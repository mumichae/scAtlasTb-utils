import warnings
from pprint import pprint


def get_from_config(
    config: dict,
    query: list,
    default: str | bool | float | int | dict | list | None = None,
    warn: bool = False,
) -> str | bool | float | int | dict | list | None:
    """Get any key from the config via query

    Args:
        config (str): config dictionary
        query (list): list of keys to walk down the config
        default (Union[str,bool,float,int,dict,list, None], optional): default value if key not found. Defaults to None.

    Returns
    -------
        Union[str,bool,float,int,dict,list, None]: value of query in config
    """
    value = config  # start at top level
    for q in query:  # walk down query
        try:
            value = value[q]
        except (AttributeError, KeyError, TypeError):
            if warn:
                warnings.warn(
                    f"key {q} not found in config for query {query}, returning default",
                    stacklevel=2,
                )
            return default
    return value


def set_defaults(config, modules=None, warn=False):
    """Set default entries for datasets in config.

    :param config: configuration dictionary with DATASETS and defaults
    :param modules: list of modules to set defaults for, e.g. ['integration', 'metrics']
                    if None, defaults for all modules are set
    :param warn: if True, warn if no defaults are defined for a dataset
    :return: updated config with defaults set for each dataset
    """
    if "defaults" not in config:
        config["defaults"] = {}
    if "datasets" not in config["defaults"]:
        config["defaults"]["datasets"] = list(config["DATASETS"].keys())

    if modules is None:
        modules = ["integration", "metrics"]
    elif isinstance(modules, str):
        modules = [modules]

    for module in modules:
        # initialise if module defaults not defined
        if module not in config["defaults"]:
            config["defaults"][module] = {}

        # update entries for each dataset
        for dataset in config["DATASETS"].keys():
            entry = _get_or_default_from_config(
                config=config["DATASETS"],
                defaults=config["defaults"],
                key=dataset,
                value=module,
                return_missing={},
                warn=warn,
            )

            # for TSV input make sure integration methods have the proper types
            # if module == 'integration' and isinstance(entry, list):
            #     # get parameters from config
            #     entry = {k: config['defaults'][module][k] for k in entry}

            # set entry in config
            config["DATASETS"][dataset][module] = entry

    return config


def _get_or_default_from_config(
    config,
    defaults,
    key,
    value,
    return_missing=None,
    dont_inherit=None,
    warn=False,
):
    """Get entry from config or return defaults if not present.

    :param config: part of the config with multiple entries of the same structure
    :param defaults: config defaults for keys with missing value
    :param key: top-level key of config dictionary
    :param value: points to entry within key
    :param return_missing: return value if no defaults for key, value
    :param dont_inherit: list of keys in value config that should not be inherited from defaults
    :param warn: warn if no defaults are defined for key, value
    :return: entry from config or defaults if not present
    """
    if dont_inherit is None:
        dont_inherit = []

    if key not in config.keys():
        print("config:")
        pprint(config)
        print(f"key: {key}, value: {value}")
        raise KeyError(f'Key "{key}" not found in config')

    config_at_value = config.get(key, {})
    if config_at_value is None:
        config_at_value = {}

    default_entry = defaults.get(value, return_missing)
    if value in dont_inherit and value in config_at_value.keys():
        default_entry = return_missing
    entry = config_at_value.get(value, default_entry)

    # update if entry is a dict
    if isinstance(entry, dict) and isinstance(default_entry, dict):
        entry = default_entry | entry

    return entry
