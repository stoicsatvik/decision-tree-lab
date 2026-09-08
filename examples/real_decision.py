"""Synthetic worked decision: choose a launch mode under uncertain demand."""
from decision_tree import binary_probability_flip, expected_value_of_perfect_information

p_high = 0.40
options = {
    "focused": (90.0, 30.0),
    "broad": (150.0, -20.0),
}

def ev(payoffs):
    return p_high * payoffs[0] + (1.0 - p_high) * payoffs[1]

preferred = max(options, key=lambda name: (ev(options[name]), name))
flip = binary_probability_flip(options["focused"], options["broad"])
evpi = expected_value_of_perfect_information((p_high, 1.0 - p_high), tuple(options.values()))
print(f"preferred={preferred}")
print(f"ev={ev(options[preferred]):.1f}")
print(f"flip_if_p_high>{flip:.6f}")
print(f"evpi={evpi:.1f}")
