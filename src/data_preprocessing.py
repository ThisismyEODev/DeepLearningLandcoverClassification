# -*- coding: utf-8 -*-
"""

@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

import argparse
from distutils.util import strtobool
import glob
from loguru import logger
import numpy as np
from pathlib import Path
import rasterio as rio
import time


def main() -> None:
    """Run module from command line."""
    logger.add(f"logs/{time.strftime('%Y%m%d_%H%M%S')}.log", retention="10 days")
    logger.info("Preprocessing EuroSAT data ...")
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Preprocessing EuroSAT data."
    )

    parser.add_argument(
        "--input",
        type=Path,
        help="Directory of the input raster files",
        required=True,
    )
    parser.add_argument(
        "--export-numpy",
        type=strtobool,
        help="Export dataset as .npy file. Default is False",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path of the output directory where to store processed data",
        required=True,
    )

    args = parser.parse_args()
    for raster_path in args.input.glob("*.tif"):
        raster_path = Path(raster_path)
        with rio.open(raster_path) as img:
            out_meta = img.meta
            out_img = np.empty((args.n_bands, img.height, img.width))
            for i in range(args.n_bands):
                band = img.read(i + 1).astype("float32")

                # Apply atmospheric correction

                # if required
                out_img[i] = band

            if args.n_bands == 1:
                out_img = out_img[0]

            out_img = out_img.astype("float32")
            out_meta["count"] = args.n_bands
            out_meta["compress"] = "deflate"

#            patches = np.array([x.name for x in patches])
#     patches_df = pd.DataFrame(patch_filenames_intersected)
#    patches_df.to_csv(dataset_path / "data.csv", index=False, header=None)

#            if args.export_numpy:
#                np.save(args.output / f"{raster_path.stem}.npy", out_img)
#            else:
#                write_raster_to_file(
#                    out_img, file_path=args.output / raster_path.name, raster_meta=out_meta
#                )


    logger.info(
        f"\n\npre-processing finished in"
        f" {(time.time() - start_time) / 60:.1f} minutes.\n"
    )


if __name__ == "__main__":
    main()

