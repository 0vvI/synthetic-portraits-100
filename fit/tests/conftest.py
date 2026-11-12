"""Put fit/ and fit/tests/ on sys.path, so the tests can import solver_ref and synth from any directory."""
import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_FIT_DIR = os.path.dirname(_TESTS_DIR)
for _path in (_TESTS_DIR, _FIT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)
