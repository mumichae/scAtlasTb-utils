import pytest

import scatlastb_utils as atl


def test_strip_barcodes():
    # make two pools of barcodes, one to pass, one to fail
    bc1 = ["ACTGACTGACTGACTG", "ACTGACTGACTGAAAA-1", "PREFIX_ACTGACTGACTGTTTT"]
    bc2 = ["this-will-brick"]
    # this is to fall out of the stripping
    assert atl.pp.strip_barcodes(bc1) == ["ACTGACTGACTGACTG", "ACTGACTGACTGAAAA", "ACTGACTGACTGTTTT"]
    # this throws a ValueError if it fails to find the barcode
    with pytest.raises(ValueError):
        atl.pp.strip_barcodes(bc2)


def test_find_library_obs():
    import pandas as pd

    # let's build a proof of concept obs for testing
    obs = pd.DataFrame.from_dict(
        {
            "ACTGACTGACTGACTG": ["A1", "filler"],
            "ACTGACTGACTGAAAA-1": ["A1", "filler"],
            "PREFIX_ACTGACTGACTGTTTT": ["A1", "filler"],
            "ACTGACTGACTGTTTT:suffix": ["B2", "filler"],
        },
        orient="index",
        columns=["library", "filler"],
    )
    # make two pools of barcodes, one to pass, one to fail
    bc1 = ["ACTGACTGACTGAAAA-different-ending", "PREFIX_ACTGACTGACTGTTTT", "missing-ACTGACTGACTGGGGG-from-pool"]
    bc2 = ["this-will-brick"]
    # this will throw a KeyError as the default library key is missing
    with pytest.raises(KeyError):
        atl.pp.find_library_obs(bc1, obs)
    # this should still throw a KeyError as the check for library key happens first
    with pytest.raises(KeyError):
        atl.pp.find_library_obs(bc2, obs)
    # this will throw a ValueError as the barcodes are bricked
    with pytest.raises(ValueError):
        atl.pp.find_library_obs(bc2, obs, library_key="library")
    # this should run
    sub, found = atl.pp.find_library_obs(bc1, obs, library_key="library")
    # these should be a dataframe and series respectively
    assert isinstance(sub, pd.DataFrame)
    assert isinstance(found, pd.Series)
    # let's check the output is correctly sized
    # we expect the entirety of A1 here, not just the overlap
    assert sub.shape == (3, 2)
    # let's check that the match is correct
    assert found["library"] == "A1"
    assert found["input_cell_count"] == 3
    assert found["library_cell_count"] == 3
    assert found["overlap"] == 2
