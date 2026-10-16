import numpy as np

from lumembrane import fit_bleach_curve


def test_monoexponential_bleach_fit_recovers_curve():
    sampling_rate = 100.0
    time_s = np.arange(2_000) / sampling_rate
    trace = 80.0 * np.exp(-time_s / 8.0) + 20.0

    fitted, params = fit_bleach_curve(trace, hz=sampling_rate, model="monoexp")

    np.testing.assert_allclose(fitted, trace, rtol=1e-6, atol=1e-6)
    assert abs(params["tau_sec"] - 8.0) < 1e-3
