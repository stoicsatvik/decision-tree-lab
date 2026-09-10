import subprocess
import sys


def test_worked_decision_is_deterministic_and_ranks_flip_capable_assumptions():
    expected = (
        "preferred=focused\n"
        "assumption=high_demand current=0.400 flip=0.454545 distance=0.054545\n"
        "assumption=high_demand_alt current=0.550 flip=0.454545 distance=0.095455\n"
        "evpi=14.4\n"
    )
    first = subprocess.run([sys.executable, "examples/real_decision.py"], check=True, capture_output=True, text=True).stdout
    second = subprocess.run([sys.executable, "examples/real_decision.py"], check=True, capture_output=True, text=True).stdout
    assert first == second == expected
    lines = first.splitlines()
    assert lines[1].startswith("assumption=high_demand ")
    assert lines[2].startswith("assumption=high_demand_alt ")
