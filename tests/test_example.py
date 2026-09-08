import subprocess
import sys


def test_worked_decision_is_deterministic_and_actionable():
    expected = "preferred=focused\nev=54.0\nflip_if_p_high>0.454545\nevpi=14.4\n"
    first = subprocess.run([sys.executable, "examples/real_decision.py"], check=True, capture_output=True, text=True).stdout
    second = subprocess.run([sys.executable, "examples/real_decision.py"], check=True, capture_output=True, text=True).stdout
    assert first == second == expected
