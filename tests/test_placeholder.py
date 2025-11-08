import raptors


def test_array_placeholder_executes():
    # The placeholder `array` function currently returns `None`, but the call
    # should succeed without raising exceptions.
    assert raptors.array() is None

