import numpy as np

from slscan import geometry, track


def test_rigid_fit_recovers_a_motion():
    rng = np.random.default_rng(0)
    source = rng.uniform(-0.1, 0.1, (12, 3))
    rotation = track.rodrigues(np.array([0.3, -0.2, 0.5]))
    translation = np.array([0.01, -0.4, 0.2])
    r, t = geometry.rigid_fit(source, source @ rotation.T + translation)
    assert np.allclose(r, rotation) and np.allclose(t, translation)
