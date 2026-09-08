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
