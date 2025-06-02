from pathlib import Path

from .config import set_defaults
from .ModuleConfig import InputFiles, ModuleConfig, WildcardParameters, _get_or_default_from_config


class PipelineConfig:
    """Class for overall pipeline configuration."""

    def __init__(self, config: dict):
        """Initialize pipeline with config.

        :param config: config dict
        """
        self.config = config

    def update_input_files_per_dataset(
        self,
        dataset: str,
        module_name: str,
        config: dict,
        first_module: str = None,
        config_class_map: dict[str:ModuleConfig] = None,
        config_kwargs: dict[str:dict] = None,
    ) -> dict:
        """
        Update input files for a given dataset and module.

        :param dataset: dataset name
        :param module_name: module name
        :param first_module: starting module name
        :param config_class_map: mapping of module names to ModuleConfig (or inherited) classes, in cases where there are module-specific config classes
        :param config_kwargs: kwargs for initializing ModuleConfig classes
        """
        if module_name == first_module:
            raise ValueError(f"Circle detected: first module {first_module} cannot be an input module")
        if first_module is None:
            first_module = module_name
        if config_class_map is None:
            config_class_map = {}
        if config_kwargs is None:
            config_kwargs = {}

        file_map = InputFiles.parse(self.config["DATASETS"][dataset]["input"][module_name])
        input_files = {}
        for file_name, file_path in file_map.items():
            if "," in file_path and ".zarr" not in file_path and ".h5ad" not in file_path:
                # multiple files
                file_paths = file_path.split(",")
            else:
                file_paths = [file_path]

            for file_path in file_paths:
                file_path = file_path.strip()
                if "/" in file_path:
                    # assert Path(file_path).exists(), f'Missing input file "{file_path}"'
                    input_files |= {file_name: file_path}
                    continue

                # get output files for input module
                input_module = file_path  # rename for easier readability
                self.config = self.update_input_files_per_dataset(
                    dataset=dataset,
                    module_name=input_module,
                    config=self.config,
                    first_module=first_module,
                    config_class_map=config_class_map,
                )

                ModuleConfigClass = config_class_map.get(input_module, ModuleConfig)
                input_cfg = ModuleConfigClass(
                    module_name=input_module,
                    config=self.config,
                    warn=False,
                    **config_kwargs.get(input_module, {}),
                )
                output_files = input_cfg.get_output_files(subset_dict={"dataset": dataset}, as_dict=True)
                input_files |= InputFiles.parse(output_files)

        self.config["DATASETS"][dataset]["input"][module_name] = input_files

    def update_file_for_module_param(
        self,
        dataset: str,
        module_name: str,
        key: str,
        subset_dict: dict = None,
        config_class_map: dict[str:ModuleConfig] = None,
        config_kwargs: dict[str:dict] = None,
    ):
        """Update a file path in the config for a specific module parameter.

        This function checks if the file pattern for a module parameter is in the input files of the dataset.

        :param dataset: dataset name
        :param module_name: module name
        :param config: config dict
        :param key: key in module that should specify a file name
        :param subset_dict: dictionary with subset parameters, e.g. {"dataset": "dataset_name"}
        :param config_class_map: mapping of module names to ModuleConfig (or inherited) classes, in cases where there are module-specific config classes
        :param config_kwargs: kwargs for initializing ModuleConfig classes
        """
        if config_class_map is None:
            config_class_map = {}
        if config_kwargs is None:
            config_kwargs = {}

        dataset_config = self.config["DATASETS"].get(dataset, {})
        file_pattern = dataset_config.get(module_name, {}).get(key)
        if file_pattern in dataset_config["input"]:
            input_module = file_pattern
            ModuleConfigClass = config_class_map.get(input_module, ModuleConfig)
            input_cfg = ModuleConfigClass(
                module_name=input_module,
                config=self.config,
                **config_kwargs.get(input_module, {}),
            )
            if subset_dict is None:
                subset_dict = {}
            subset_dict |= {"dataset": dataset}
            dataset_config[module_name][key] = input_cfg.get_output_files(subset_dict=subset_dict)[0]

        # set updated dataset config
        self.config["DATASETS"][dataset] = dataset_config
