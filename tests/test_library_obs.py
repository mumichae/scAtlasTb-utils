import pandas as pd
import pytest

import scatlastb_utils as atl


# this defines a simple dummy obs data frame for testing
# can use obs as a positional argument for tests and it will pull in this
# (that's what the fixture part is for)
@pytest.fixture
def obs():
    return pd.DataFrame.from_dict(
        {
            "ACTGACTGACTGACTG": ["A1", "filler"],
            "ACTGACTGACTGAAAA-1": ["A1", "filler"],
            "PREFIX_ACTGACTGACTGTTTT": ["A1", "filler"],
            "ACTGACTGACTGTTTT:suffix": ["B2", "filler"],
        },
        orient="index",
        columns=["library", "filler"],
    )


# this is the syntax for testing a bunch of different combinations
# need to provide the argument names, matching the actual test function below
# and then a variety of inputs
@pytest.mark.parametrize(
    "barcodes,expected,raises",
    [
        pytest.param(
            ["ACTGACTGACTGACTG", "ACTGACTGACTGAAAA-1", "PREFIX_ACTGACTGACTGTTTT"],
            ["ACTGACTGACTGACTG", "ACTGACTGACTGAAAA", "ACTGACTGACTGTTTT"],
            None,
            id="valid barcodes",
        ),
        pytest.param(
            ["this-will-brick"],
            None,
            ValueError,
            id="invalid barcode",
        ),
    ],
)
def test_strip_barcodes(barcodes, expected, raises):
    """Test atl.pp.strip_barcodes for valid and invalid inputs."""
    # the syntax for checking for an exception and checking the actual output are different
    # so if the expected exception is not None, go in and handle it exception style
    if raises:
        with pytest.raises(raises):
            atl.pp.strip_barcodes(barcodes)
    else:
        # and if it is none, check an assert of the output
        assert atl.pp.strip_barcodes(barcodes) == expected


# in the interest of legibility, the actual tests for this function are split into two
# in this part, we're checking for exceptions
# also note the appearance of the obs fixture from earlier in the arguments
@pytest.mark.parametrize(
    "barcodes,library_key,expected_exception",
    [
        # KeyError: default library key missing
        pytest.param(
            ["ACTGACTGACTGAAAA-different-ending", "PREFIX_ACTGACTGACTGTTTT", "missing-ACTGACTGACTGGGGG-from-pool"],
            None,
            KeyError,
            id="missing default library key (good barcodes)",
        ),
        # KeyError: default library key missing (should be checked before the barcodes)
        pytest.param(
            ["this-will-brick"],
            None,
            KeyError,
            id="missing default library key (bad barcodes)",
        ),
        # ValueError: bricked barcodes
        pytest.param(
            ["this-will-brick"],
            "library",
            ValueError,
            id="invalid barcode with library key",
        ),
    ],
)
def test_find_library_obs_exceptions(barcodes, library_key, expected_exception, obs):
    if library_key:
        with pytest.raises(expected_exception):
            atl.pp.find_library_obs(barcodes, obs, library_key=library_key)
    else:
        with pytest.raises(expected_exception):
            atl.pp.find_library_obs(barcodes, obs)


# here we're checking everything runs as expected
@pytest.mark.parametrize(
    "barcodes,library_key,expected_shape,expected_found",
    [
        # Success case
        pytest.param(
            ["ACTGACTGACTGAAAA-different-ending", "PREFIX_ACTGACTGACTGTTTT", "missing-ACTGACTGACTGGGGG-from-pool"],
            "library",
            (3, 2),
            pd.Series({"library": "A1", "input_cell_count": 3, "library_cell_count": 3, "overlap": 2}),
            id="valid input with library key",
        ),
    ],
)
def test_find_library_obs(barcodes, library_key, expected_shape, expected_found, obs):
    sub, found = atl.pp.find_library_obs(barcodes, obs, library_key=library_key)
    assert isinstance(sub, pd.DataFrame)
    assert isinstance(found, pd.Series)
    assert sub.shape == expected_shape
    # can handily check whether two series are identical like so
    pd.testing.assert_series_equal(found, expected_found)
