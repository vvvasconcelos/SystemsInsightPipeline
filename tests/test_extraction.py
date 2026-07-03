import numpy as np
import pandas as pd
import pytest

from sip_systemsinsightpipeline.cld import Extract


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


EXPECTED_ADJACENCY = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, -1, 0],
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
    extract.adjacency_matrix_from_kumu()

    assert np.array_equal(extract.adjacency_matrix, EXPECTED_ADJACENCY)
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
