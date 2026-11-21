"""Unit tests for the custom-equation parser and evaluator (sip_systemsinsightpipeline.equations)."""
import numpy as np
import pandas as pd
import pytest

from sip_systemsinsightpipeline.equations import (EquationEvaluator, EquationParser,
                                                  parse_equations_from_dataframe)


def parse(variables, equation, incoming=None, var="Y", var_types=None):
    incoming = [v for v in variables if v != var] if incoming is None else incoming
    return EquationParser(variables, var_types=var_types).parse_equation(var, equation, incoming)


def evaluate(r, vals, params=None, intervention=0.0, t=0.0, var="Y"):
    return EquationEvaluator({var: r}).evaluate(var, vals, params or {}, intervention, t)


def errors(r):
    return [w for w in r["validation_warnings"] if w.startswith("ERROR")]


# ----------------------------------------------------------------------------- variable names
@pytest.mark.parametrize("variables, equation, used, vals, expected", [
    (["X", "X dev", "Y"], "X dev - X", {"X", "X dev"}, {"X": 1.0, "X dev": 3.0}, 2.0),
    (["Price", "Price index", "Y"], "Price index * Price", {"Price", "Price index"},
     {"Price": 2.0, "Price index": 5.0}, 10.0),
    (["Level", "Level change", "Y"], "Level change + 0 * Level", {"Level", "Level change"},
     {"Level": 7.0, "Level change": 1.5}, 1.5),
    (["A", "AA", "Y"], "AA + A", {"A", "AA"}, {"A": 1.0, "AA": 10.0}, 11.0),
])
def test_whole_names_longest_first(variables, equation, used, vals, expected):
    r = parse(variables, equation)
    assert r["is_valid"] and r["variables_used"] == used
    assert evaluate(r, vals) == pytest.approx(expected)


@pytest.mark.parametrize("name", [
    "Relative price difference (plant vs. animal)",
    "Farmer's income",
    'Quoted "name"',
    "CO2-emissions",
    "2nd wave",
    "Cost $ per unit",
    "Product #1 sales",
    "Share (%)",
    "a.b.c",
])
def test_names_with_punctuation(name):
    r = parse([name, "Y"], f"2 * {name} + 1")
    assert r["is_valid"], r["validation_warnings"]
    assert r["variables_used"] == {name}
    assert evaluate(r, {name: 3.0}) == pytest.approx(7.0)


def test_whitespace_in_equations_is_normalised():
    r = parse(["Price index", "Y"], "2 *  Price\n index")
    assert r["variables_used"] == {"Price index"}
    assert evaluate(r, {"Price index": 2.0}) == pytest.approx(4.0)


def test_name_is_not_matched_inside_numbers_parameters_or_attributes():
    r = parse(["e", "rate", "np", "Y"], "1e-3 * e + #rate * rate + np.exp(np)")
    assert r["is_valid"], r["validation_warnings"]
    assert r["variables_used"] == {"e", "rate", "np"}
    assert r["named_parameters"] == ["rate"]
    assert evaluate(r, {"e": 1000.0, "rate": 2.0, "np": 0.0}, {"#rate": 3.0}) == pytest.approx(1 + 6 + 1)


# ----------------------------------------------------------------------------- unknown names (I-02)
def test_typo_is_rejected_with_suggestion():
    r = parse(["Habit", "Intention", "Y"], "#1 * Habit + #2 * Intenton")
    assert not r["is_valid"]
    (err,) = errors(r)
    assert "'Intenton'" in err and "did you mean 'Intention'" in err


def test_multi_word_unknown_phrase_is_reported_whole():
    r = parse(["Habit strength", "Y"], "Habbit strength + 1")
    assert not r["is_valid"]
    assert "'Habbit strength'" in errors(r)[0] and "Habit strength" in errors(r)[0]


@pytest.mark.parametrize("equation, hint", [
    ("p1 * A", "did you mean '#1'"),
    ("A * (time > 2)", "written t"),
    ("min(A, 1)", "np.minimum"),
    ("pi * A", "np.pi"),
])
def test_unknown_name_hints(equation, hint):
    r = parse(["A", "Y"], equation)
    assert not r["is_valid"] and hint in errors(r)[0]


def test_single_letter_unknown_names_are_errors():
    assert not parse(["A", "Y"], "A * x")["is_valid"]


def test_number_literals_are_not_names():
    r = parse(["A", "Y"], "1e-3 * A + 2.5E2 + .5")
    assert r["is_valid"] and evaluate(r, {"A": 1000.0}) == pytest.approx(1 + 250 + 0.5)


def test_reserved_placeholder_names_are_rejected():
    assert not parse(["A", "Y"], "A + _v0_")["is_valid"]


# ----------------------------------------------------------------------------- functions (I-13)
def test_bare_function_names_get_numpy_prefix():
    r = parse(["A", "Y"], "exp(A) + tanh(0) + maximum(A, 2)")
    assert "np.exp(" in r["compiled_equation"] and "np.maximum(" in r["compiled_equation"]
    assert evaluate(r, {"A": 0.0}) == pytest.approx(1.0 + 0.0 + 2.0)


@pytest.mark.parametrize("equation", ["sigmoid(A)", "np.sigmoid(A)"])
def test_sigmoid(equation):
    r = parse(["A", "Y"], equation)
    assert r["is_valid"] and evaluate(r, {"A": 1.0}) == pytest.approx(1 / (1 + np.exp(-1.0)))
    assert evaluate(r, {"A": -1000.0}) == pytest.approx(0.0)        # no overflow


def test_numpy_function_checks():
    assert not parse(["A", "Y"], "np.foo(A)")["is_valid"]           # does not exist
    assert not parse(["A", "Y"], "np.random(A)")["is_valid"]        # not a function
    r = parse(["A", "Y"], "np.cbrt(A)")                             # exists, not in the list
    assert r["is_valid"] and any("Unknown numpy function" in w for w in r["validation_warnings"])
    assert evaluate(r, {"A": 8.0}) == pytest.approx(2.0)


def test_variable_used_like_a_function_is_rejected():
    r = parse(["A", "B", "Y"], "A (B)")
    assert not r["is_valid"] and "followed by '('" in errors(r)[0]


def test_function_without_call_is_rejected():
    assert not parse(["A", "Y"], "A + exp")["is_valid"]


@pytest.mark.parametrize("equation", [
    "A.real", "().__class__", "(lambda: 1)()", "'text'", "[A][0]", "np.__dict__",
])
def test_unsupported_syntax_is_rejected(equation):
    assert not parse(["A", "Y"], equation)["is_valid"]


def test_comparisons_and_conditional_expressions():
    r = parse(["S", "Y"], "(S < 0.5) * 1.0 - (S >= 0.5) * 1.0 + (2 if S > 1 else 3)")
    assert evaluate(r, {"S": 0.0}) == pytest.approx(1 + 3)
    assert evaluate(r, {"S": 2.0}) == pytest.approx(-1 + 2)


# ----------------------------------------------------------------------------- parameters (I-01, I-09)
@pytest.mark.parametrize("equation, indices", [
    ("# * A + # * B", [1, 2]),
    ("#1 * A + # * B", [1, 2]),
    ("# * A + #1 * B", [1, 2]),
    ("#2 * A + # * B", [2, 3]),
    ("#1 * A + #3 * B", [1, 3]),
    ("#10 * A + #1 * B", [1, 10]),
])
def test_local_parameter_numbering(equation, indices):
    r = parse(["A", "B", "Y"], equation)
    assert r["parameter_indices"] == indices and r["n_parameters"] == len(indices)
    params = {f"#{i}": float(i) for i in indices}
    assert np.isfinite(evaluate(r, {"A": 1.0, "B": 1.0}, params))


def test_named_parameters_are_parsed_and_compiled():
    r = parse(["Price", "Y"], "#Price * Price + #k_2 + #1")
    assert r["named_parameters"] == ["Price", "k_2"] and r["parameter_indices"] == [1]
    assert evaluate(r, {"Price": 2.0}, {"#Price": 3.0, "#k_2": 1.0, "#1": 0.5}) == pytest.approx(7.5)


def test_missing_parameter_value_gives_clear_error():
    r = parse(["A", "Y"], "#rate * A")
    with pytest.raises(ValueError, match="#rate"):
        evaluate(r, {"A": 1.0}, {})


# ----------------------------------------------------------------------------- intervention and time (I-07)
def test_intervention_placeholder():
    r = parse(["A", "Y"], "#1 * ($ + A)")
    assert r["has_intervention"]
    assert evaluate(r, {"A": 1.0}, {"#1": 2.0}, intervention=0.5) == pytest.approx(3.0)
    r2 = parse(["A", "Y"], "#1 * A")                 # no $: the intervention is added
    assert evaluate(r2, {"A": 1.0}, {"#1": 2.0}, intervention=0.5) == pytest.approx(2.5)
    r3 = parse(["A", "Y"], "$ + $ * A")
    assert any("Multiple $" in w for w in r3["validation_warnings"])


def test_time_variable():
    r = parse(["K", "Y"], "K * (t > 2) + 0.1 * t")
    assert r["uses_time"] and r["is_valid"]
    assert evaluate(r, {"K": 1.0}, t=1.0) == pytest.approx(0.1)
    assert evaluate(r, {"K": 1.0}, t=3.0) == pytest.approx(1.3)


def test_model_variable_named_t_shadows_time():
    r = parse(["t", "Y"], "2 * t")
    assert not r["uses_time"] and r["variables_used"] == {"t"}
    assert evaluate(r, {"t": 5.0}, t=100.0) == pytest.approx(10.0)


# ----------------------------------------------------------------------------- diagram checks (I-17)
def test_link_warnings():
    r = parse(["A", "B", "C", "Y"], "A + B", incoming=["A", "C"])
    msgs = " ".join(r["validation_warnings"])
    assert "{'B'} used in equation but not defined as incoming links" in msgs
    assert "Incoming link(s) {'C'}" in msgs
    assert r["is_valid"]


def test_self_reference_depends_on_variable_type():
    types = {"S": "stock", "K": "constant"}
    r = parse(["S", "K"], "(1 - S) * 0.5 + K", incoming=["K"], var="S", var_types=types)
    assert r["is_valid"] and r["validation_warnings"] == []
    r = parse(["S", "K"], "0.5 * S + K", incoming=["K"], var="S", var_types={"S": "auxiliary"})
    assert not r["is_valid"] and "algebraic loop" in errors(r)[0]
    r = parse(["S", "K"], "0.5 * S + K", incoming=["K"], var="S")   # type unknown: warn only
    assert r["is_valid"] and any("{'S'}" in w for w in r["validation_warnings"])


# ----------------------------------------------------------------------------- misc
@pytest.mark.parametrize("equation", [None, np.nan, "", "   ", "nan"])
def test_empty_equations_are_not_valid(equation):
    assert not parse(["A", "Y"], equation)["is_valid"]


def test_numeric_equation_cell():
    r = parse(["Y"], 1, incoming=[])        # an Excel cell holding the number 1
    assert r["is_valid"] and evaluate(r, {}) == pytest.approx(1.0)


def test_parse_equations_from_dataframe_helper():
    df = pd.DataFrame({"Label": ["Y", "A"], "Equation": ["2 * A", np.nan]})
    adj = pd.DataFrame([[0, 1], [0, 0]], index=["Y", "A"], columns=["Y", "A"])
    eqs, msgs = parse_equations_from_dataframe(df, ["Y", "A"], adj, ["Y", "A"])
    assert set(eqs) == {"Y"} and msgs == []
