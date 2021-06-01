# -*- coding: utf-8 -*-
"""
Main file of the deep learning based landcover classification challenge.

For more information, see project's GitLab repo:

    https://github.com/ThisismyEODev/DeepLearningLandcoverClassification.git


@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

import argparse
import time

from loguru import logger

import parameter_file as parameters
from .folder_setup import setup_workspace
from .data_download import retrieve_data


def main() -> None:

    """Run module from command line."""

    logger.add(f"logs/{time.strftime('%Y%m%d_%H%M%S')}.log", retention="10 days")
    logger.info("Starting landcover classification ...")
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="UP 42 Coding Challenge."
    )

    print("Setting up the workspace")
    setup_workspace()

    print("Access data")


    
if __name__ == "__main__":
    main()

