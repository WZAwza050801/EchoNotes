"""Run the pipeline2 test suite without pytest: python run_tests.py

The tests import the real package path (work.pipeline2.*), so discovery runs
from the repository root. pytest users can also run `pytest work/pipeline2/tests/`.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    suite = unittest.TestLoader().discover(
        str(ROOT / "work" / "pipeline2" / "tests"), top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
