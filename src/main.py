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


def main() -> None:

    """Run module from command line."""

    logger.add(f"logs/{time.strftime('%Y%m%d_%H%M%S')}.log", retention="10 days")
    logger.info("Starting landcover classification ...")
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="UP 42 Coding Challenge."
    )

    parser.add_argument(
        "--dir",
        type=str,
        help="Path to working directory we want to work in.",
        required=False,
    )
    args = parser.parse_args()
    working_dir = args.dir

    logger.info("Setting up the directories ...")
    setup_workspace(working_dir)

    # Pre process one folder at a time

    # Predict

    
if __name__ == "__main__":
    main()

