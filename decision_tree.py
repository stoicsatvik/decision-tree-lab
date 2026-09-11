"""Deterministic decision analysis primitives. Standard library only."""
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass(frozen=True)
class Outcome:
    name: str
    payoff: float


@dataclass(frozen=True)
class Chance:
    name: str
    branches: Tuple[Tuple[float, "Node"], ...]


@dataclass(frozen=True)
class Choice:
    name: str
    options: Tuple["Node", ...]


Node = Union[Outcome, Chance, Choice]


@dataclass(frozen=True)
class Summary:
    name: str
    expected_value: float
    worst_case: float


@dataclass(frozen=True)
class ProbabilitySensitivity:
    state: str
    current_probability: float
    flip_probability: float
    distance_to_flip: float
    preferred_below: str
    preferred_above: str


@dataclass(frozen=True)
class PayoffSensitivity:
    option: str
    state_index: int
    current_payoff: float
    flip_payoff: float
    distance_to_flip: float


@dataclass(frozen=True)
class FlipAssumption:
    kind: str
    target: str
    current_value: float
    flip_value: float
    normalized_distance: float


def _validate_probability(p: float) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError("probability must be in [0, 1]")


def evaluate(node: Node) -> Summary:
    if isinstance(node, Outcome):
        return Summary(node.name, float(node.payoff), float(node.payoff))
    if isinstance(node, Chance):
        if not node.branches:
            raise ValueError("chance node requires branches")
        total = sum(p for p, _ in node.branches)
        for p, _ in node.branches:
            _validate_probability(p)
        if abs(total - 1.0) > 1e-12:
            raise ValueError("chance probabilities must sum to 1")
        children = [(p, evaluate(child)) for p, child in node.branches]
        return Summary(node.name, sum(p * s.expected_value for p, s in children), min(s.worst_case for _, s in children))
    if isinstance(node, Choice):
        if not node.options:
            raise ValueError("choice node requires options")
        ranked = sorted((evaluate(option) for option in node.options), key=lambda s: (-s.expected_value, -s.worst_case, s.name))
        best = ranked[0]
        return Summary(best.name, best.expected_value, best.worst_case)
    raise TypeError("unsupported node type")


def preferred_option(choice: Choice) -> Summary:
    return evaluate(choice)


def binary_probability_flip(payoff_a: Tuple[float, float], payoff_b: Tuple[float, float]) -> float | None:
    """Probability p where p*a0+(1-p)*a1 equals p*b0+(1-p)*b1."""
    numerator = payoff_b[1] - payoff_a[1]
    denominator = (payoff_a[0] - payoff_a[1]) - (payoff_b[0] - payoff_b[1])
    if abs(denominator) <= 1e-15:
        return None
    p = numerator / denominator
    return p if 0.0 <= p <= 1.0 else None


def binary_probability_sensitivity(state: str, current_probability: float, option_a: Tuple[str, Tuple[float, float]], option_b: Tuple[str, Tuple[float, float]]) -> ProbabilitySensitivity | None:
    """Return the nearest binary probability threshold capable of flipping two options."""
    _validate_probability(current_probability)
    name_a, payoff_a = option_a
    name_b, payoff_b = option_b
    flip = binary_probability_flip(payoff_a, payoff_b)
    if flip is None:
        return None

    def preferred(p: float) -> str:
        ev_a = p * payoff_a[0] + (1.0 - p) * payoff_a[1]
        ev_b = p * payoff_b[0] + (1.0 - p) * payoff_b[1]
        if abs(ev_a - ev_b) <= 1e-12:
            return min(name_a, name_b)
        return name_a if ev_a > ev_b else name_b

    epsilon = min(1e-9, max(flip, 1.0 - flip) * 1e-9)
    below = preferred(max(0.0, flip - epsilon))
    above = preferred(min(1.0, flip + epsilon))
    if below == above:
        return None
    return ProbabilitySensitivity(state, current_probability, flip, abs(current_probability - flip), below, above)


def payoff_flip_sensitivity(probabilities: Tuple[float, ...], option: Tuple[str, Tuple[float, ...]], competitor: Tuple[str, Tuple[float, ...]], state_index: int) -> PayoffSensitivity | None:
    """Payoff threshold in one state where two options tie, holding all else fixed."""
    if not probabilities:
        raise ValueError("probabilities must be non-empty")
    for p in probabilities:
        _validate_probability(p)
    if abs(sum(probabilities) - 1.0) > 1e-12:
        raise ValueError("probabilities must sum to 1")
    name, payoffs = option
    _, competitor_payoffs = competitor
    if len(payoffs) != len(probabilities) or len(competitor_payoffs) != len(probabilities):
        raise ValueError("every option must define one payoff per state")
    if not 0 <= state_index < len(probabilities):
        raise IndexError("state_index out of range")
    coefficient = probabilities[state_index]
    if coefficient <= 1e-15:
        return None
    competitor_ev = sum(p * x for p, x in zip(probabilities, competitor_payoffs))
    fixed_ev = sum(p * x for i, (p, x) in enumerate(zip(probabilities, payoffs)) if i != state_index)
    flip = (competitor_ev - fixed_ev) / coefficient
    current = float(payoffs[state_index])
    return PayoffSensitivity(name, state_index, current, flip, abs(current - flip))


def assumption_flip_report(state_names: Tuple[str, str], probabilities: Tuple[float, float], option_a: Tuple[str, Tuple[float, float]], option_b: Tuple[str, Tuple[float, float]]) -> Tuple[FlipAssumption, ...]:
    """Rank single-assumption changes that can flip a binary decision.

    Probability distance is normalized to its [0,1] domain. Payoff distance is
    normalized by the largest absolute payoff in the decision, making unlike
    assumptions comparable without pretending they share physical units.
    """
    if len(state_names) != 2:
        raise ValueError("binary report requires exactly two state names")
    if len(probabilities) != 2:
        raise ValueError("binary report requires exactly two probabilities")
    scale = max(abs(x) for _, payoffs in (option_a, option_b) for x in payoffs)
    if scale <= 1e-15:
        raise ValueError("payoff scale must be non-zero")
    rows = []
    probability = binary_probability_sensitivity(state_names[0], probabilities[0], option_a, option_b)
    if probability is not None:
        rows.append(FlipAssumption("probability", state_names[0], probability.current_probability, probability.flip_probability, probability.distance_to_flip))
    for option, competitor in ((option_a, option_b), (option_b, option_a)):
        for index, state in enumerate(state_names):
            payoff = payoff_flip_sensitivity(probabilities, option, competitor, index)
            if payoff is not None:
                rows.append(FlipAssumption("payoff", f"{option[0]}:{state}", payoff.current_payoff, payoff.flip_payoff, payoff.distance_to_flip / scale))
    return tuple(sorted(rows, key=lambda row: (row.normalized_distance, row.kind, row.target)))


def expected_value_of_perfect_information(probabilities: Tuple[float, ...], option_payoffs: Tuple[Tuple[float, ...], ...]) -> float:
    """EVPI for discrete states when the state is observed before choosing an option."""
    if not probabilities or not option_payoffs:
        raise ValueError("probabilities and options must be non-empty")
    states = len(probabilities)
    if any(len(payoffs) != states for payoffs in option_payoffs):
        raise ValueError("every option must define one payoff per state")
    for p in probabilities:
        _validate_probability(p)
    if abs(sum(probabilities) - 1.0) > 1e-12:
        raise ValueError("probabilities must sum to 1")
    without_information = max(sum(p * payoff for p, payoff in zip(probabilities, payoffs)) for payoffs in option_payoffs)
    with_information = sum(p * max(payoffs[state] for payoffs in option_payoffs) for state, p in enumerate(probabilities))
    return max(0.0, with_information - without_information)
