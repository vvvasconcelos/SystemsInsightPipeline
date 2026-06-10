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
