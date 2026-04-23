import hashlib

import pandas as pd

# Re-exported for backwards compatibility — canonical definitions live in utils


def get_use_gpu(config):
    """TODO: move to ModuleConfig?"""
    use_gpu = bool(config.get("use_gpu", False))
    if isinstance(use_gpu, str):
        use_gpu = use_gpu.lower() == "true"
    return use_gpu


def all_but(_list, is_not):
    """Returns a list with all elements except the specified one."""
    return [x for x in _list if x != is_not]


def unique_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from a pandas DataFrame.

    :param df: pandas DataFrame
    :return: DataFrame with unique rows
    """
    if df.empty:
        return df
    # hashable_columns = [
    #     col for col in df.columns
    #     if all(isinstance(df[col].iloc[i], typing.Hashable) for i in range(df.shape[0]))
    # ]
    # duplicated = df[hashable_columns].duplicated()
    duplicated = df.astype(str).duplicated()
    return df[~duplicated].reset_index(drop=True)


def expand_dict(_dict: dict) -> zip:
    """Create a cross-product on a dictionary with literals and lists

    :param _dict: dictionary with lists and literals as values
    :return: zip of wildcards and dictionaries
    """
    df = pd.DataFrame({k: [v] if isinstance(v, list) else [[v]] for k, v in _dict.items()})
    for col in df.columns:
        df = df.explode(col)
    dict_list = df.apply(lambda row: dict(zip(df.columns, row, strict=False)), axis=1)

    def remove_chars(s, chars="{} ',"):
        for c in chars:
            s = s.replace(c, "")
        return s

    wildcards = df.apply(
        lambda row: "-".join([remove_chars(f"{col[0]}:{x}") for col, x in zip(df.columns, row, strict=False)]), axis=1
    )
    return zip(wildcards, dict_list, strict=False)


def expand_dict_and_serialize(_dict: dict, do_not_expand: list = None) -> zip:
    """
    Create a cross-product on a dictionary with literals and lists

    :param _dict: dictionary with lists and literals as values
    :return: list of dictionaries with literals as values
    """
    import hashlib

    import jsonpickle

    if do_not_expand is None:
        do_not_expand = []

    df = pd.DataFrame({k: [v] if isinstance(v, list) and k not in do_not_expand else [[v]] for k, v in _dict.items()})
    for col in df.columns:
        df = df.explode(col)
    dict_list = df.apply(lambda row: dict(zip(df.columns, row, strict=False)), axis=1)

    wildcards = [hashlib.blake2b(jsonpickle.encode(d).encode("utf-8"), digest_size=5).hexdigest() for d in dict_list]

    return zip(wildcards, dict_list, strict=False)


def unlist_dict(_dict: dict) -> dict:
    """Unlist a dictionary, converting single-item lists to their values."""
    return {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in _dict.items()}


def unpack_dict_in_df(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Given a column in a pandas dataframe containing dictionaries, extract these to top level.

    :param df: pandas dataframe
    :param column: column name containing dictionaries
    """
    return df.drop(columns=column).assign(**df[column].dropna().apply(pd.Series, dtype=object))


def ifelse(statement, _if, _else):
    """Single-line if-else wrapper.

    :param statement: Condition to evaluate
    :param _if: Value to return if the condition is True
    :param _else: Value to return if the condition is False
    """
    if statement:
        return _if
    else:
        return _else


def merge(dfs: list, verbose: bool = True, **kwargs):
    """Merge list of dataframes

    :param dfs: list of dataframes
    :param kwargs: arguments passed to pd.merge
    :return: merged dataframe
    """
    from functools import reduce

    merged_df = reduce(lambda x, y: pd.merge(x, y, **kwargs), dfs)
    if verbose:
        print(merged_df)
    return merged_df


def create_hash(string: str, digest_size: int = 5):
    """Create a unique hash from a string using BLAKE2b hashing algorithm."""
    string = string.encode("utf-8")
    return hashlib.blake2b(string, digest_size=digest_size).hexdigest()
