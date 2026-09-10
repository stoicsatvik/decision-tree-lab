from decision_tree import Chance, Choice, Outcome, binary_probability_flip, binary_probability_sensitivity, evaluate, expected_value_of_perfect_information


def test_chance_ev_and_downside():
    node = Chance("launch", ((0.6, Outcome("win", 100)), (0.4, Outcome("lose", -20))))
    result = evaluate(node)
    assert result.expected_value == 52.0
    assert result.worst_case == -20.0


def test_choice_prefers_higher_ev():
    safe = Outcome("safe", 40)
    risky = Chance("risky", ((0.5, Outcome("up", 120)), (0.5, Outcome("down", -10))))
    result = evaluate(Choice("root", (safe, risky)))
    assert result.name == "risky"
    assert result.expected_value == 55.0
    assert result.worst_case == -10.0


def test_choice_tie_break_is_deterministic():
    result = evaluate(Choice("root", (Outcome("z", 10), Outcome("a", 10))))
    assert result.name == "a"


def test_probability_flip_threshold():
    p = binary_probability_flip((100, -20), (40, 40))
    assert p is not None
    assert abs(p - 0.5) < 1e-12


def test_probability_sensitivity_reports_flip_direction_and_distance():
    result = binary_probability_sensitivity("high demand", 0.40, ("focused", (90, 30)), ("broad", (150, -20)))
    assert result is not None
    assert abs(result.flip_probability - 5.0 / 11.0) < 1e-12
    assert abs(result.distance_to_flip - (5.0 / 11.0 - 0.40)) < 1e-12
    assert result.preferred_below == "focused"
    assert result.preferred_above == "broad"


def test_probability_sensitivity_returns_none_when_no_flip_exists():
    result = binary_probability_sensitivity("demand", 0.5, ("dominant", (20, 10)), ("inferior", (5, 0)))
    assert result is None


def test_probability_sensitivity_rejects_invalid_current_probability():
    try:
        binary_probability_sensitivity("demand", 1.1, ("a", (1, 0)), ("b", (0, 1)))
    except ValueError as exc:
        assert "[0, 1]" in str(exc)
    else:
        raise AssertionError("invalid current probability accepted")


def test_invalid_probability_mass_fails_closed():
    try:
        evaluate(Chance("bad", ((0.8, Outcome("a", 1)), (0.3, Outcome("b", 2)))))
    except ValueError as exc:
        assert "sum to 1" in str(exc)
    else:
        raise AssertionError("invalid probability mass accepted")


def test_empty_choice_fails_closed():
    try:
        evaluate(Choice("bad", ()))
    except ValueError as exc:
        assert "requires options" in str(exc)
    else:
        raise AssertionError("empty choice accepted")


def test_perfect_information_has_positive_value_when_state_changes_choice():
    value = expected_value_of_perfect_information((0.5, 0.5), ((100, 0), (40, 40)))
    assert value == 20.0


def test_perfect_information_zero_when_same_option_dominates_every_state():
    value = expected_value_of_perfect_information((0.25, 0.75), ((10, 20), (5, 15)))
    assert value == 0.0


def test_perfect_information_rejects_mismatched_states():
    try:
        expected_value_of_perfect_information((0.5, 0.5), ((10,), (0, 20)))
    except ValueError as exc:
        assert "one payoff per state" in str(exc)
    else:
        raise AssertionError("mismatched state vectors accepted")
