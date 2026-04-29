# Copyright 2024 Bloomberg Finance L.P.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


from typing import List, Dict, Tuple
from tqdm.auto import tqdm
import torch
import numpy as np

from .utils import get_top_k_pages, get_top_k_pages_single_page_from_each_doc

class RAGModelBase:
    def __init__(
        self,
        retrieval_model=None,
        qa_model=None,
        vqa_model=None,
    ):
        """Base class for RAG pipeline
        
        - retrieval_model: arbitrary retrieval model (e.g., ColPali / ColBERT)
        - qa_model: arbitrary text-only QA model (e.g., LLama3)
        - vqa_model: arbitrary VQA model (e.g., InternVL2, GPT4-o)
        
        """
        self.retrieval_model = retrieval_model
        self.qa_model = qa_model
        self.vqa_model = vqa_model

        if self.retrieval_model is not None:
            self.retrieval_model.model.eval()
        if self.vqa_model is not None:
            self.vqa_model.model.eval()
        if self.qa_model is not None:
            self.qa_model.model.eval()
    
    
    @staticmethod
    def _page_uid_to_doc_page(page_uid: str) -> Tuple[str, int]:
        doc_id = page_uid.split("_page")[0]
        page_idx = int(page_uid.split("_page")[-1])
        return doc_id, page_idx

    def _retrieve_pages_from_index_query_meta(
        self,
        query_meta,
        index,
        token2pageuid,
        all_token_embeddings,
        n_return_pages: int,
        ignore_pad_scores_in_final_ranking: bool = False,
    ) -> List[Tuple]:
        query_emb = query_meta["embeddings"].float().numpy().astype(np.float32)
        raw_tokens = query_meta.get("raw_tokens", [])
        score_active_query_token_mask = None
        if ignore_pad_scores_in_final_ranking:
            score_active_query_token_mask = [token != "<pad>" for token in raw_tokens]
            if score_active_query_token_mask and not any(score_active_query_token_mask):
                raise ValueError(
                    "Ignoring PAD scores in final ranking removed every scoring query token."
                )

        k = n_return_pages
        _distances, indices = index.search(query_emb, k)

        final_page2scores = {}
        for q_idx, current_query_emb in enumerate(query_emb):
            current_q_page2scores = {}

            for nn_idx in range(k):
                found_nearest_doc_token_idx = indices[q_idx, nn_idx]
                page_uid = token2pageuid[found_nearest_doc_token_idx]

                doc_token_emb = all_token_embeddings[found_nearest_doc_token_idx]
                score = (current_query_emb * doc_token_emb).sum()

                if page_uid not in current_q_page2scores:
                    current_q_page2scores[page_uid] = score
                else:
                    current_q_page2scores[page_uid] = max(current_q_page2scores[page_uid], score)

            if score_active_query_token_mask is not None and not score_active_query_token_mask[q_idx]:
                continue

            for page_uid, score in current_q_page2scores.items():
                if page_uid in final_page2scores:
                    final_page2scores[page_uid] += score
                else:
                    final_page2scores[page_uid] = score

        sorted_pages = sorted(final_page2scores.items(), key=lambda x: x[1], reverse=True)
        top_k_pages = sorted_pages[:k]

        sorted_results = []
        for page_uid, score in top_k_pages:
            doc_id, page_idx = self._page_uid_to_doc_page(page_uid)
            sorted_results.append((doc_id, page_idx, score.item()))

        return sorted_results

    def _filter_query_embeddings_for_scoring(
        self,
        query_meta,
        ignore_pad_scores_in_final_ranking: bool = False,
    ) -> torch.Tensor:
        filtered_query_emb = query_meta["embeddings"]
        if not ignore_pad_scores_in_final_ranking:
            return filtered_query_emb

        raw_tokens = query_meta.get("raw_tokens", [])
        score_keep_mask = torch.tensor([token != "<pad>" for token in raw_tokens], dtype=torch.bool)
        if not score_keep_mask.any():
            raise ValueError(
                "Ignoring PAD scores in final ranking removed every scoring query token."
            )
        return filtered_query_emb[score_keep_mask]

    def _rerank_candidate_pages_exact(
        self,
        candidate_page_uids: List[str],
        query_embeds: torch.Tensor,
        docid2embs: Dict[str, torch.Tensor],
        n_return_pages: int,
        single_page_from_each_doc: bool = False,
        rerank_batch_size: int = 128,
        show_progress: bool = False,
    ) -> List[Tuple]:
        if rerank_batch_size <= 0:
            raise ValueError("rerank_batch_size must be a positive integer.")

        scored_pages = []
        starts = range(0, len(candidate_page_uids), rerank_batch_size)
        if show_progress:
            starts = tqdm(starts, desc="Reranking candidate pages", total=(len(candidate_page_uids) + rerank_batch_size - 1) // rerank_batch_size)

        for start_idx in starts:
            batch_page_uids = candidate_page_uids[start_idx:start_idx + rerank_batch_size]
            batch_doc_embeds = []
            for page_uid in batch_page_uids:
                doc_id, page_idx = self._page_uid_to_doc_page(page_uid)
                page_embeds = docid2embs[doc_id][page_idx]
                page_embeds = page_embeds.view(-1, page_embeds.shape[-1])
                batch_doc_embeds.append(page_embeds)

            batch_scores = self.retrieval_model.retrieve(
                query=None,
                doc_embeds=batch_doc_embeds,
                query_embeds=[query_embeds],
                to_cpu=True,
                return_top_1=False,
            )
            batch_scores = batch_scores.flatten().tolist()
            scored_pages.extend(zip(batch_page_uids, batch_scores))

        scored_pages.sort(key=lambda x: x[1], reverse=True)
        if single_page_from_each_doc:
            best_pages = []
            seen_doc_ids = set()
            for page_uid, score in scored_pages:
                doc_id, page_idx = self._page_uid_to_doc_page(page_uid)
                if doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)
                best_pages.append((doc_id, page_idx, float(score)))
                if len(best_pages) >= n_return_pages:
                    break
            return best_pages

        reranked_results = []
        for page_uid, score in scored_pages[:n_return_pages]:
            doc_id, page_idx = self._page_uid_to_doc_page(page_uid)
            reranked_results.append((doc_id, page_idx, float(score)))
        return reranked_results

    def retrieve_pages_from_docs(
        self,
        query: str,
        docid2embs: Dict[str, torch.Tensor],
        docid2lens: Dict[str, torch.Tensor] = None,
        
        index = None,
        token2pageuid = None,
        all_token_embeddings = None,

        n_return_pages: int = 1,
        single_page_from_each_doc: bool = False,
        query_token_filter: str = "full",
        ignore_pad_scores_in_final_ranking: bool = False,
        candidate_n_pages: int = None,
        candidate_query_token_filter: str = "full",
        candidate_rerank_batch_size: int = 128,
        show_progress=False,
    ) -> List[Tuple]:
        """
        Given text query and pre-extracted document embedding,
        calculate similarity scores and return top-n pages

        Args:
            - query (str): a text query to call retrieval model  
            - docid2embs (Dict[str, tensor]): collection of document embeddings
                key: document_id
                value: torch.tensor of size (n_tokens, emb_dim)
            - index: faiss index
            - n_return_pages (int): number of pages to return
            - single_page_from_each_doc (bool): if true, only single page is retrieved from each PDF document.
            - query_token_filter (str): token filter used in the final scoring/reranking stage.
            - ignore_pad_scores_in_final_ranking (bool): if true, keep PAD in query encoding/search but exclude PAD
              contributions when computing the final page score.
            - candidate_n_pages (int): optional two-stage candidate pool size. If set, stage 1 builds a shared
              candidate page pool of this size with candidate_query_token_filter, then stage 2 reranks only those
              pages using query_token_filter / ignore_pad_scores_in_final_ranking.
            - candidate_query_token_filter (str): token filter used only for stage-1 candidate generation.

        Return:
            retrieval_results
            [(doc_id, page_idx, scores)...]
        """


        if index is not None:
            if candidate_n_pages is not None and candidate_n_pages > 0:
                if candidate_n_pages < n_return_pages:
                    raise ValueError(
                        "candidate_n_pages must be >= n_return_pages for two-stage retrieval."
                    )
                stage1_query_meta = self.retrieval_model.encode_query_with_metadata(
                    query=query,
                    to_cpu=True,
                    query_token_filter=candidate_query_token_filter,
                )
                candidate_results = self._retrieve_pages_from_index_query_meta(
                    query_meta=stage1_query_meta,
                    index=index,
                    token2pageuid=token2pageuid,
                    all_token_embeddings=all_token_embeddings,
                    n_return_pages=candidate_n_pages,
                    ignore_pad_scores_in_final_ranking=False,
                )
                candidate_page_uids = [
                    f"{doc_id}_page{page_idx}" for doc_id, page_idx, _score in candidate_results
                ]

                stage2_query_meta = self.retrieval_model.encode_query_with_metadata(
                    query=query,
                    to_cpu=True,
                    query_token_filter=query_token_filter,
                )
                stage2_query_embeds = self._filter_query_embeddings_for_scoring(
                    stage2_query_meta,
                    ignore_pad_scores_in_final_ranking=ignore_pad_scores_in_final_ranking,
                )
                return self._rerank_candidate_pages_exact(
                    candidate_page_uids=candidate_page_uids,
                    query_embeds=stage2_query_embeds,
                    docid2embs=docid2embs,
                    n_return_pages=n_return_pages,
                    single_page_from_each_doc=single_page_from_each_doc,
                    rerank_batch_size=candidate_rerank_batch_size,
                    show_progress=show_progress,
                )

            query_meta = self.retrieval_model.encode_query_with_metadata(
                query=query,
                to_cpu=True,
                query_token_filter=query_token_filter,
            )
            return self._retrieve_pages_from_index_query_meta(
                query_meta=query_meta,
                index=index,
                token2pageuid=token2pageuid,
                all_token_embeddings=all_token_embeddings,
                n_return_pages=n_return_pages,
                ignore_pad_scores_in_final_ranking=ignore_pad_scores_in_final_ranking,
            )

        query_meta = self.retrieval_model.encode_query_with_metadata(
            query=query,
            to_cpu=True,
            query_token_filter=query_token_filter,
        )
        filtered_query_emb = self._filter_query_embeddings_for_scoring(
            query_meta,
            ignore_pad_scores_in_final_ranking=ignore_pad_scores_in_final_ranking,
        )

        docid2scores = {}
        for doc_id, doc_embs in tqdm(
            docid2embs.items(),
            total=len(docid2embs),
            disable = not show_progress,
            desc=f"Calculating similarity score over documents"
        ):
            doc_lens = None
            if docid2lens is not None:
                doc_lens = docid2lens[doc_id]

            scores = self.retrieval_model.retrieve(
                query=None,
                doc_embeds=doc_embs,
                doc_lens=doc_lens,
                query_embeds=[filtered_query_emb],
                to_cpu=True,
                return_top_1=False
            )
            scores = scores.flatten().tolist()
            docid2scores[doc_id] = scores

        # find the pages with top scores
        if single_page_from_each_doc:
            return get_top_k_pages_single_page_from_each_doc(docid2scores=docid2scores, k=n_return_pages)
        else:
            return get_top_k_pages(docid2scores=docid2scores, k=n_return_pages)

        
    def run_qa(self):
        raise NotImplementedError

    def run_vqa(self):
        raise NotImplementedError
