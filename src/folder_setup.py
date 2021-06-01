# -*- coding: utf-8 -*-
"""


@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

from pathlib import Path
import parameter_file as parameters

def setup_workspace(path):
    """
    Sets up the directories for this challenge

    Parameters
    ----------
    path : str or Path
        path to where we want to work.

    """
    if isinstance(path, str):
        path = Path(path)
    elif isinstance(path, Path):
        path = path
    print("The Setup will occur here: ", str(path))

    data_path = path / 'inputdata'
    if data_path.is_dir() is False:
        print("Create the data directory")
        data_path.mkdir()

    model_path = path / 'model_directory'
    if model_path.is_dir() is False:
        print("Create the directory for storing models")
        model_path.mkdir()


def main() -> None:
    setup_workspace(parameters.path)


if __name__ == "__main__":
    main()
