# -*- coding: utf-8 -*-
"""
Useful function for visualization
"""
from typing import Union

from rich.console import Console
from rich.table import Table


def print_tables(title: str, headers: Union[list, tuple], contents: Union[list, tuple]):
    """
    Print a table in console

    Parameters
    ----------
    title: str
        Title of table.
    headers: Union[list, tuple]
        Header of table.
    contents: Union[list, tuple]
        Content of table.
    """
    # Create table
    table = Table(title=title)

    # add headers
    for header in headers:
        table.add_column(header)

    # add contents
    for content in contents:
        content = list(map(str, content))
        table.add_row(*content)

    # print table
    console = Console()
    console.print(table)


def print_best_params(study):
    """
    Print the best parameters of a study.
    """
    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    trial_with_highest_psnr = max(study.best_trials, key=lambda t: t.values[0])
    print("Trial with highest PSNR: ")
    print_tables(
        "Trial with highest PSNR",
        headers=tuple(trial_with_highest_psnr.params.keys()),
        contents=[tuple(trial_with_highest_psnr.params.values())],
    )
