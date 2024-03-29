# -*- coding: utf-8 -*-
"""
Script for downloading data.
"""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import join
from typing import Dict

from loguru import logger
from requests import ConnectionError, ConnectTimeout, ReadTimeout, get
from rich.progress import track

logger.add(sys.stderr, format="{time}-{level}: {message}")


def download_image(image_path: str, url_image: str, suffix: str) -> str:
    """
    Download image.

    Parameters
    ----------
    image_path : str
        Image path.
    url_image : str
        Image URL.
    suffix : str
        Filename suffix.

    Returns
    -------
    str
        Image name.
    """
    extension = url_image.split("/")[-1].split(".")[-1]
    response = get(url_image, timeout=60, allow_redirects=True)
    with open(join(image_path, f"image-{suffix}.{extension}"), mode="wb") as file:
        file.write(response.content)
    return f"image-{suffix}.{extension}"


def download_set(data_url: str, data_path: str) -> None:
    """
    Download set of data.

    Parameters
    ----------
    data_url : str
        Data URL
    data_path : str
        Where data will be stored.
    """
    data: Dict = get(data_url, timeout=60).json()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for row in data.get("rows", []):
            futures.append(
                executor.submit(
                    download_image,
                    image_path=data_path,
                    url_image=row["row"]["grayscale_image"]["src"],
                    suffix=row["row_idx"],
                )
            )
        for future in as_completed(futures):
            try:
                future.result()
            except (ConnectTimeout, ConnectionError, ReadTimeout):
                logger.warning("Error during ")


def download_data(data_url: str, data_path: str, size: int) -> None:
    """
    Download the necessary data.

    Parameters
    ----------
    data_url : str
        Data URL
    data_path : str
        Where data will be stored.
    size : int
        Size of the data.
    """
    # ##: Download images.
    for offset in track(range(0, size + 1, 100), description="[green]Download images ..."):
        download_set(data_url=data_url + f"&offset={offset}&length=100", data_path=data_path)


if __name__ == "__main__":
    import os

    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())
    download_data(data_url=os.environ.get("TRAIN_URL"), data_path=join(os.environ.get("RAW_PATH"), "train"), size=10000)
    download_data(data_url=os.environ.get("TESTS_URL"), data_path=join(os.environ.get("RAW_PATH"), "tests"), size=6000)
