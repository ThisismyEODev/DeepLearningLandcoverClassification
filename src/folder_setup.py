# -*- coding: utf-8 -*-
"""


@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

from pathlib import Path


def setup_workspace(path):
    """
    Sets up the directories for this challenge

    Parameters
    ----------
    path : str or Path
        path to where we want to work.

    Returns
    -------
    None.

    """
    if isinstance(path, str):
        path = Path(path)

    data_path = path / 'inputdata'
    # if data_path is empty... download

    model_path = path / 'model'
    results_path = path / 'results'


