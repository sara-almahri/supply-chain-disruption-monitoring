"""The items of the repeated runs (Appendix C.6): a random 10% of the test items, drawn within each item type.

    python -m evaluation.benchmark_v2.repeat_sample

Draws 10% of the test items of each type, rounded to the nearest item, so that the subset keeps the composition of the
test set. Each type is drawn with its own generator, seeded by 20261014 and the type. Writes
items/test/repeat_sample.json. The primary system is then run five further times on these items
(run_pipeline --repeat 2 to 6).
"""
import argparse
import json
import random

from evaluation.benchmark_v2.test_set import in_set
from evaluation.benchmark_v2.specify import OUT

FOLDER = OUT / "test"
SEED = 20261014
TYPES = ("P", "N1", "N2", "N3", "N4")


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    specs = {s["item_id"]: s for s in json.loads((FOLDER / "specs.json").read_text())["items"]}
    kept = in_set()
    sample = {"seed": SEED, "share": 0.10, "by_type": {}}
    for t in TYPES:
        pool = sorted(i for i in kept if specs[i]["item_type"] == t)
        sample["by_type"][t] = sorted(random.Random(f"{SEED}-{t}").sample(pool, round(0.10 * len(pool))))
    sample["items"] = sorted(i for v in sample["by_type"].values() for i in v)
    (FOLDER / "repeat_sample.json").write_text(json.dumps(sample, indent=1))
    print({t: len(v) for t, v in sample["by_type"].items()}, len(sample["items"]))


if __name__ == "__main__":
    main()
