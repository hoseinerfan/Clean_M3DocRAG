# HPC Setup Notes

This repository was set up and validated on the UNC cluster under:

- repo root: `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG`
- conda env: `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/env`
- data root: `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/data`
- model root: `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/model`
- embeddings root: `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/embeddings`
- output root: `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output`

## Validated Stack

Upstream `pyproject.toml` is only partially pinned. The following resolved stack was needed to make the Bloomberg code, `colpali-engine==0.3.1`, and Qwen2-VL work together on this cluster:

- Python `3.10.20`
- CUDA toolkit module `cuda/12.1.1`
- `torch==2.2.2+cu121`
- `torchvision==0.17.2+cu121`
- `transformers==4.49.0`
- `tokenizers==0.21.4`
- `flash-attn==2.5.8` built from source against CUDA 12.1.1
- `numpy==1.26.4`
- `fsspec==2026.2.0`

The exact realized environment was frozen on the HPC clone with:

```bash
pip freeze > requirements.lock.txt
```

## Data Layout

The runtime loader expects the following layout:

```text
data/m3-docvqa/
  multimodalqa/
    MMQA_dev.jsonl
    MMQA_train.jsonl
    MMQA_texts.jsonl
    MMQA_images.jsonl
    MMQA_tables.jsonl
  id_url_mapping.jsonl
  dev_doc_ids.json
  train_doc_ids.json
  splits/
    pdfs_dev/
```

Two practical deviations from the upstream docs were needed on this cluster:

1. The MMQA downloader in `m3docvqa` hit cluster SSL trust issues, so the exact MMQA files were fetched from the official `allenai/multimodalqa` GitHub repo via sparse checkout instead.
2. `playwright` itself was installed, but Chromium had to be installed explicitly with `PLAYWRIGHT_BROWSERS_PATH` pointing inside the repo tree.

## Generated Artifacts

Validated dev-set artifacts:

- dev PDFs downloaded: `3366`
- corrupted PDFs: `0`
- embedding directory: `embeddings/colpali-v1.2_m3-docvqa_dev`
- FAISS index: `embeddings/colpali-v1.2_m3-docvqa_dev_pageindex_ivfflat/index.bin`

Observed sizes:

- embeddings directory: about `24G`
- FAISS index: about `23G`

The indexing script is CPU/RAM bound and loads all embeddings into memory before flattening them. On the validated run it consumed comfortably less than the available RAM on a `503 GiB` node.

## Full Dev Result

Validated full dev run:

- retrieval model: `colpaligemma-3b-pt-448-base` + `colpali-v1.2`
- VQA model: `Qwen2-VL-7B-Instruct`
- FAISS index type: `ivfflat`
- retrieval pages: `1`
- precision: `16-bit`

Reported metrics:

- recall@1: `0.47589395446830934`
- list EM: `27.038099139696847`
- list F1: `31.250716919295368`

Output files were written under `output/rag_dev/`.

## Known Gotchas

- In `examples/run_page_embedding.py`, `--data_len` does not limit work when `--loop_unique_doc_ids=True`, because the dataset length is driven by the full supporting-doc-id list in that code path.
- `examples/run_rag_m3docvqa.py` still loads all embeddings and rebuilds the flattened token array even for small smoke tests.
- The `m3docvqa` split generator writes `dev_doc_ids.json` and `train_doc_ids.json` relative to the current working directory, so it should be run from `data/m3-docvqa/` if you want the files in the final runtime location.

## Publishing The Exact HPC Lockfile

This repository copy does not contain the exact HPC-generated `requirements.lock.txt`, because that file lives on the HPC clone. After pulling the latest repo changes on the HPC clone, run:

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
./scripts/stage_hpc_artifacts.sh
git commit -m "Add HPC lockfile and Slurm pipeline notes"
git push
```

That stages:

- `requirements.lock.txt`
- `HPC_SETUP.md`
- `slurm/m3docrag_dev_pipeline.sh`
- `.gitignore`
