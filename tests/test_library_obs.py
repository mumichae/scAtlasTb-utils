import pandas as pd
import pytest

import scatlastb_utils as atl


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
    if raises:
        with pytest.raises(raises):
            atl.pp.strip_barcodes(barcodes)
    else:
        assert atl.pp.strip_barcodes(barcodes) == expected


@pytest.mark.parametrize(
    "barcodes,library_key,expected_exception,expected_shape,expected_found",
    [
        # KeyError: default library key missing
        (
            ["ACTGACTGACTGAAAA-different-ending", "PREFIX_ACTGACTGACTGTTTT", "missing-ACTGACTGACTGGGGG-from-pool"],
            None,
            KeyError,
            None,
            None,
        ),
        # KeyError: default library key missing
        (["this-will-brick"], None, KeyError, None, None),
        # ValueError: bricked barcodes
        (["this-will-brick"], "library", ValueError, None, None),
        # Success case
        (
            ["ACTGACTGACTGAAAA-different-ending", "PREFIX_ACTGACTGACTGTTTT", "missing-ACTGACTGACTGGGGG-from-pool"],
            "library",
            None,
            (3, 2),
            {"library": "A1", "input_cell_count": 3, "library_cell_count": 3, "overlap": 2},
        ),
    ],
    ids=[
        "missing default library key (bc1)",
        "missing default library key (bc2)",
        "invalid barcode with library key",
        "valid input with library key",
    ],
)
def test_find_library_obs(barcodes, library_key, expected_exception, expected_shape, expected_found, obs):
    if expected_exception:
        if library_key:
            with pytest.raises(expected_exception):
                atl.pp.find_library_obs(barcodes, obs, library_key=library_key)
        else:
            with pytest.raises(expected_exception):
                atl.pp.find_library_obs(barcodes, obs)
    else:
        sub, found = atl.pp.find_library_obs(barcodes, obs, library_key=library_key)
        assert isinstance(sub, pd.DataFrame)
        assert isinstance(found, pd.Series)
        assert sub.shape == expected_shape
        for k, v in expected_found.items():
            assert found[k] == v
