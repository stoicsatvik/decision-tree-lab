"""Synthetic worked decision: rank assumptions by ability to flip the choice."""
from decision_tree import binary_probability_sensitivity, expected_value_of_perfect_information

options = {"focused": (90.0, 30.0), "broad": (150.0, -20.0)}
assumptions = {"high_demand": 0.40, "high_demand_alt": 0.55}

def ev(payoffs, p):
    return p * payoffs[0] + (1.0 - p) * payoffs[1]

p = assumptions["high_demand"]
preferred = max(options, key=lambda name: (ev(options[name], p), name))
sensitivities = [binary_probability_sensitivity(name, value, ("focused", options["focused"]), ("broad", options["broad"])) for name, value in assumptions.items()]
ranked = sorted((s for s in sensitivities if s is not None), key=lambda s: (s.distance_to_flip, s.state))
evpi = expected_value_of_perfect_information((p, 1.0 - p), tuple(options.values()))
print(f"preferred={preferred}")
for s in ranked:
    print(f"assumption={s.state} current={s.current_probability:.3f} flip={s.flip_probability:.6f} distance={s.distance_to_flip:.6f}")
print(f"evpi={evpi:.1f}")
