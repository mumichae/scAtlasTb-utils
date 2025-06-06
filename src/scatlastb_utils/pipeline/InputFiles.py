from pathlib import Path

import pandas as pd

from .config import get_from_config
from .misc import create_hash


class InputFiles:
    """Class to handle input files for a specific module.

    This class is designed to parse input files from a configuration dictionary,
    map them to unique identifiers, and provide easy access to these files across
    different datasets. It also supports writing the file mapping to a specified
    output directory.
    """

    def __init__(
        self,
        module_name: str,
        dataset_config: dict,
        output_directory: [str, Path] = None,
    ):
        """Initialize InputFiles.

        :param module_name: name of the module, needed for querying the config
        :param dataset_config: config containing dataset name, inputs and module parameters
        :param output_directory: directory to write the file mapping to, if None, no file is written
        """
        self.module_name = module_name
        self.config = dataset_config
        self.out_dir = output_directory

        self.file_map = {}
        self.wildcards = dict(dataset=[], file_id=[], file_name=[])
        for dataset in self.config:
            self.set_file_per_dataset(dataset)
        self.file_map_df = pd.DataFrame(
            [
                (dataset, name, file_path)
                for dataset, name_map in self.file_map.items()
                for name, file_path in name_map.items()
            ],
            columns=["dataset", "file_id", "file_path"],
        )

        if self.out_dir is not None and self.file_map_df.shape[0] > 0:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            # write file mapping to file
            self.file_map_df.to_csv(self.out_dir / "input_files.tsv", sep="\t", index=False)

    @staticmethod
    def parse(input_files: [str, list, dict], digest_size: int = 5) -> dict:
        """Parse input files.

        Given input files, convert into file name to file path mapping.
        If no file names are provided, create a unique hash code.
        """
        if isinstance(input_files, str):
            input_files = [input_files]

        if isinstance(input_files, list):
            input_files = {create_hash(file): file for file in input_files}

        if not isinstance(input_files, dict):
            raise ValueError(f"input_files must be a list or dict, but is {type(input_files)}")

        return input_files

    def set_file_per_dataset(self, dataset: str, digest_size: int = 5):
        """Set input files for a given dataset.

        This function maps an input file with its unique identifier, if no file ID is specified

        :param dataset: dataset key in config['DATASETS']
        """
        input_files = get_from_config(config=self.config[dataset], query=["input", self.module_name])
        input_files = self.parse(input_files, digest_size=digest_size)
        self.file_map[dataset] = input_files

        # Reshape the file ID to dataset mapping to a lists of wildcards
        for file_id, file in input_files.items():
            self.wildcards["dataset"].append(dataset)
            self.wildcards["file_id"].append(file_id)
            self.wildcards["file_name"].append(file)

    def get_wildcards(self) -> dict:
        """Get input filename wildcards for all datasets."""
        return self.wildcards

    def get_files(self, as_df=False) -> [dict, pd.DataFrame]:
        """Get the file name to file path mapping for all datasets."""
        return self.file_map_df if as_df else self.file_map

    def get_files_per_dataset(self, dataset) -> dict:
        """Get file name to file path mapping for a given dataset."""
        try:
            file = self.file_map[dataset]
        except KeyError:
            raise KeyError(f'No files found for dataset "{dataset}" file_map: {self.file_map}') from None
        return file

    def get_file(self, dataset, file_id) -> [str, Path]:
        """Get file path for a given dataset and file ID."""
        return self.get_files_per_dataset(dataset)[file_id]
