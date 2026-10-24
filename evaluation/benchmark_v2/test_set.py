"""The items of the relevance benchmark: items/test/test_set.json, the 1,117 items that passed the checks and the audit."""
import json

from evaluation.benchmark_v2.specify import OUT


def in_set():
    """Item ids of the test set, sorted."""
    return json.loads((OUT / "test" / "test_set.json").read_text())["items"]
