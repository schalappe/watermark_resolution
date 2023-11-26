# -*- coding: utf-8 -*-
"""
Script for downloading data.
"""
import sys
from os.path import join
from rich.progress import track
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests import get, ConnectTimeout, ConnectionError, ReadTimeout
from loguru import logger


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
    data = get(data_url, timeout=60).json()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for row in data.get("rows", []):
            futures.append(
                executor.submit(
                    download_image,
                    image_path=join(data_path, "images"),
                    url_image=row["row"]["grayscale_image"]["src"],
                    suffix=row["row_idx"],
                )
            )
        for future in as_completed(futures):
            try:
                future.result()
            except (ConnectTimeout, ConnectionError, ReadTimeout):
                logger.warning("Error during ")


def download_data(data_url: str, data_path: str) -> None:
    """
    Download the necessary data.

    Parameters
    ----------
    data_url : str
        Data URL
    data_path : str
        Where data will be stored.
    """
    # ##: Download images.
    for offset in track(range(0, 10001, 100), description="[green]Download images ..."):
        download_set(data_url=data_url + f"&offset={offset}&length=100", data_path=data_path)


if __name__ == "__main__":
    import os
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())
    download_data(data_url=os.environ.get("DATA_URL"), data_path=os.environ.get("RAW_PATH"))
