import numpy as np
import pandas as pd
import pytest

from sip_systemsinsightpipeline.cld import Extract
from sip_test_helpers import NAN, quiet, write_workbook as make_workbook


def write_workbook(path, element_types):
    """Write a 3-variable example workbook (A -> B -> C -> A plus two interactions)."""
    df_e = pd.DataFrame({
        "Label": ["A", "B", "C"],
        "Type": element_types,
        "Tags": [0, -1, 1],
        "Description": ["VOI", "", ""],
    })
    df_c = pd.DataFrame({
        "From": ["A", "B", "C"],
        "Type": ["+", "-", "+"],
        "To": ["B", "C", "A"],
    })
    df_i = pd.DataFrame({
        "From1": ["A", "B"],
        "From2": ["C", "C"],
        "Type": ["+", "+"],
        "To": ["B", "A"],
    })
    with pd.ExcelWriter(path) as writer:
        df_e.to_excel(writer, sheet_name="Elements", index=False)
        df_c.to_excel(writer, sheet_name="Connections", index=False)
        df_i.to_excel(writer, sheet_name="Interactions", index=False)


# C is a constant: its incoming link B -> C is ignored (constants are exogenous), so its row
# is zero, with a warning (issue I-15).
EXPECTED_ADJACENCY = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 0, 0],
])

EXPECTED_INTERACTIONS = np.array([
    [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
])


def test_extraction_matrices(tmp_path):
    """Port of the former inline Extract.test_extraction."""
    xlsx = tmp_path / "evidence_table.xlsx"
    write_workbook(xlsx, ["stock", "auxiliary", "constant"])

    extract = Extract(str(xlsx))
    with pytest.warns(UserWarning, match="Constant 'C' has 1 incoming link"):
        extract.adjacency_matrix_from_kumu()

    assert np.array_equal(extract.adjacency_matrix, EXPECTED_ADJACENCY)
    assert np.array_equal(extract.df_adj.to_numpy(), EXPECTED_ADJACENCY)
    assert np.array_equal(extract.interactions_matrix, EXPECTED_INTERACTIONS)
    assert sorted(extract.variables) == ["A", "B", "C"]
    assert extract.variable_of_interest == ["A"]
    assert sorted(extract.intervention_variables) == ["B", "C"]


def _write_simple(path, df_e, df_c=None):
    if df_c is None:
        df_c = pd.DataFrame({"From": ["Lever"], "Type": ["+"], "To": ["S"]})
    with pd.ExcelWriter(path) as w:
        df_e.to_excel(w, sheet_name="Elements", index=False)
        df_c.to_excel(w, sheet_name="Connections", index=False)


def test_blank_tags_is_not_an_intervention(tmp_path):
    """Regression: a blank (NaN) Tags cell used to make the variable an intervention with
    NaN strength, silently corrupting every trajectory to NaN."""
    df_e = pd.DataFrame({
        "Label": ["S", "Lever", "Context"],
        "Type": ["stock", "constant", "constant"],
        "Tags": [0, 1, None],                       # Context left blank
        "Description": ["VOI", "", ""],
    })
    df_c = pd.DataFrame({"From": ["Lever", "Context"], "Type": ["+", "+"], "To": ["S", "S"]})
    _write_simple(tmp_path / "m.xlsx", df_e, df_c)
    s = Extract(str(tmp_path / "m.xlsx")).extract_settings()
    assert s.intervention_variables == ["Lever"]
    assert s.intervention_strengths["Context"] == 0.0
    assert all(np.isfinite(v) for v in s.intervention_strengths.values())


def test_text_tags_warns_and_is_not_an_intervention(tmp_path):
    """Regression: leftover Kumu tag text used to become a string 'strength' and crash
    numpy far downstream; now it is coerced to 0 with a clear warning."""
    df_e = pd.DataFrame({
        "Label": ["S", "Lever", "Context"],
        "Type": ["stock", "constant", "constant"],
        "Tags": [0, 1, "policy lever"],
        "Description": ["VOI", "", ""],
    })
    df_c = pd.DataFrame({"From": ["Lever", "Context"], "Type": ["+", "+"], "To": ["S", "S"]})
    _write_simple(tmp_path / "m.xlsx", df_e, df_c)
    with pytest.warns(UserWarning, match="Non-numeric Tags"):
        s = Extract(str(tmp_path / "m.xlsx")).extract_settings()
    assert s.intervention_variables == ["Lever"]
    assert s.intervention_strengths["Context"] == 0.0


def test_missing_description_column_warns_not_crashes(tmp_path):
    df_e = pd.DataFrame({
        "Label": ["S", "Lever"],
        "Type": ["stock", "constant"],
        "Tags": [0, 1],
    })
    _write_simple(tmp_path / "m.xlsx", df_e)
    with pytest.warns(UserWarning, match="Description"):
        s = Extract(str(tmp_path / "m.xlsx")).extract_settings()
    assert s.variable_of_interest == []


def test_missing_tags_column_gives_clear_error(tmp_path):
    df_e = pd.DataFrame({
        "Label": ["S", "Lever"],
        "Type": ["stock", "constant"],
        "Description": ["VOI", ""],
    })
    _write_simple(tmp_path / "m.xlsx", df_e)
    with pytest.warns(UserWarning, match="Tags"):
        with pytest.raises(Exception, match="at least one intervention"):
            Extract(str(tmp_path / "m.xlsx")).extract_settings()


@pytest.mark.parametrize("types", [
    ["stock", "auxiliary", "constant"],
    ["Stock", "Auxiliary", "Constant"],
    ["STOCK", "AUXILIARY", "CONSTANT"],
])
def test_extract_settings_type_case_insensitive(tmp_path, types):
    """Regression: capitalized Type values used to leave stocks_and_auxiliaries empty,
    silently producing a parameterless model."""
    xlsx = tmp_path / "evidence_table.xlsx"
    write_workbook(xlsx, types)

    s = Extract(str(xlsx)).extract_settings()

    assert s.stocks == ["A"]
    assert s.auxiliaries == ["B"]
    assert s.constants == ["C"]
    assert s.stocks_and_auxiliaries == ["A", "B"]
    assert s.stocks_and_constants == ["A", "C"]


# ----------------------------------------------------------------------------- equations at extraction
def _eq_model(tmp_path, equation, parameters=None):
    return make_workbook(tmp_path / "eq.xlsx",
                          {"Label": ["S", "K"], "Type": ["stock", "constant"], "Tags": [0, 1],
                           "Description": ["VOI", ""], "Equation": [equation, NAN]},
                          [("K", "+", "S")], parameters=parameters)


def test_invalid_equation_raises_with_the_offending_name(tmp_path):
    with pytest.raises(ValueError, match="Kk"):
        quiet(Extract(_eq_model(tmp_path, "2 * Kk")).extract_settings)


def test_variable_without_links_needs_an_equation(tmp_path):
    """I-08: a clock (equation '1') needs no incoming link; without an equation it still does."""
    df_e = pd.DataFrame({"Label": ["Clock", "S", "K"], "Type": ["stock", "stock", "constant"],
                         "Tags": [0, 0, 1], "Description": ["", "VOI", ""], "Equation": ["1", NAN, NAN]})
    links = [("K", "+", "S"), ("Clock", "+", "S")]
    s, _, _ = quiet(Extract(make_workbook(tmp_path / "a.xlsx", df_e, links)).extract_settings)
    assert "Clock" in s.equations
    df_e.loc[0, "Equation"] = NAN
    with pytest.raises(Exception, match="Clock"):
        quiet(Extract(make_workbook(tmp_path / "b.xlsx", df_e, links)).extract_settings)


def test_equation_of_a_constant_is_ignored_with_a_warning(tmp_path):
    wb = make_workbook(tmp_path / "c.xlsx",
                        {"Label": ["S", "K"], "Type": ["stock", "constant"], "Tags": [0, 1],
                         "Description": ["VOI", ""], "Equation": [NAN, "2"]}, [("K", "+", "S")])
    s, _, warns = quiet(Extract(wb).extract_settings)
    assert any("Equation of constant 'K' is ignored" in w for w in warns)


def test_empty_label_gives_clear_error(tmp_path):
    df_e = pd.DataFrame({"Label": ["S", None], "Type": ["stock", "constant"], "Tags": [0, 1],
                         "Description": ["VOI", ""]})
    with pytest.raises(Exception, match="empty Label"):
        quiet(Extract(make_workbook(tmp_path / "m.xlsx", df_e, [("S", "+", "S")])).extract_settings)


# ----------------------------------------------------------------------------- Parameters sheet
def test_parameters_sheet_is_read(tmp_path):
    wb = _eq_model(tmp_path, "#rate * K - #decay * S",
                   parameters=[("rate", 0.5, 1.5), ("#decay", 0.1, 0.1), (NAN, NAN, NAN)])
    s, out, _ = quiet(Extract(wb).extract_settings)
    assert s.parameter_priors == {"rate": (0.5, 1.5), "decay": (0.1, 0.1)}
    assert "#rate in [0.5, 1.5]" in out


@pytest.mark.parametrize("rows, message", [
    ([("1", 0, 1)], "numbered parameter"),
    ([("2rate", 0, 1)], "invalid parameter name"),
    ([("rate", 2, 1)], "Low"),
    ([("rate", "a", 1)], "numeric"),
    ([("rate", 0, 1), ("#rate", 0, 2)], "more than once"),
])
def test_parameters_sheet_validation(tmp_path, rows, message):
    wb = _eq_model(tmp_path, "#rate * K", parameters=rows)
    with pytest.raises(ValueError, match=message):
        quiet(Extract(wb).extract_settings)


def test_parameters_sheet_unused_name_warns(tmp_path):
    wb = _eq_model(tmp_path, "#rate * K", parameters=[("rate", 0, 1), ("other", 0, 1)])
    _, _, warns = quiet(Extract(wb).extract_settings)
    assert any("#other" in w and "not used" in w for w in warns)


def test_parameters_sheet_needs_its_columns(tmp_path):
    wb = _eq_model(tmp_path, "#rate * K")
    with pd.ExcelWriter(wb, mode="a", engine="openpyxl") as xw:
        pd.DataFrame({"Name": ["rate"], "Min": [0]}).to_excel(xw, sheet_name="Parameters", index=False)
    with pytest.raises(ValueError, match="Low"):
        quiet(Extract(wb).extract_settings)
