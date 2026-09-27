import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "racs_table9_audit", ROOT / "scripts/audit_racs_table9_qa.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class QuestionTextValidationTests(unittest.TestCase):
    def test_accepts_only_boundary_whitespace_differences(self):
        expected = ' Did Mike Newell  win the award BAFTA?'
        self.assertTrue(audit.same_question_except_boundary_space(expected, expected))
        self.assertTrue(audit.same_question_except_boundary_space(
            expected, 'Did Mike Newell  win the award BAFTA? '))
        self.assertFalse(audit.same_question_except_boundary_space(
            expected, 'Did Mike Newell win the award BAFTA?'))
        self.assertFalse(audit.same_question_except_boundary_space(
            expected, 'Did Mike Newell  win the award BAFTA'))
        self.assertFalse(audit.same_question_except_boundary_space(expected, None))


if __name__ == "__main__":
    unittest.main()
