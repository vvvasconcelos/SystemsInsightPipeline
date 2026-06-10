import numpy as np

from sip_systemsinsightpipeline import SDM, Extract


def build_sdm(tutorials_dir, n_samples=5):
    extract = Extract(str(tutorials_dir / "Minimal.xlsx"))
    s = extract.extract_settings()
    s.seed = 12345
    s.N = n_samples
    s.t_end = 10
    s.time_unit = "years"
    s.parameter_value_aux = 0.1
    s.parameter_value_stocks = 0.1
    return s, SDM(s)


def test_minimal_end_to_end(tutorials_dir):
    """Excel -> Extract -> SDM -> simulate -> optimize, via the SDM convenience wrapper."""
    s, sdm = build_sdm(tutorials_dir)

    df_sol_per_sample, param_samples = sdm.run_simulations()
    assert len(df_sol_per_sample) == s.N
    assert len(df_sol_per_sample[0]) == len(s.intervention_variables)

    effects = sdm.get_intervention_effects()
    voi = s.variable_of_interest[0]
    assert voi == "A"
    assert set(effects[voi].keys()) == set(s.intervention_variables)
    assert all(len(v) == s.N for v in effects[voi].values())

    params = sdm.sample_model_parameters()
    costs = [1.0] * len(s.intervention_variables)

    # The wrapper must forward keyword arguments to the optimizer unchanged
    result = sdm.optimize_intervention_intensities(
        params, costs, voi, budget=1.0, maximize=True, n_starts=2, seed=0
    )

    assert result["success"]
    assert np.isfinite(result["best_effect_size"])
    assert result["n_equilibria"] >= 1
    intensities = np.asarray(result["best_intensities"])
    assert np.all(intensities >= -1e-12)
    assert intensities @ np.asarray(costs) <= 1.0 + 1e-6


def test_minimal_simulation_reproducible(tutorials_dir):
    """Same seed, same trajectories."""
    def run():
        _, sdm = build_sdm(tutorials_dir, n_samples=3)
        df_sol, _ = sdm.run_simulations()
        return np.concatenate([np.asarray(df.values, dtype=float).ravel()
                               for sample in df_sol for df in sample])

    assert np.array_equal(run(), run())
