import pytest

from slscan import calibrate, synthetic

SPEC = calibrate.BoardSpec(9, 6, 0.04, 0.03)


@pytest.fixture(scope="session")
def spec():
    return SPEC


@pytest.fixture(scope="session")
def rig():
    return synthetic.reference_rig()


@pytest.fixture(scope="session")
def calibration(rig):
    views = synthetic.calibration_views(rig, SPEC)
    return calibrate.calibrate(views, SPEC, (rig.projector.width, rig.projector.height))
