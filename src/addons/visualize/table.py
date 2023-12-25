# -*- coding: utf-8
"""
    Useful function for visualization
"""
from typing import Union

from rich.console import Console
from rich.table import Table


def print_tables(title: str, headers: Union[list, tuple], contents: Union[list, tuple]) -> None:
    """
    Print a table in console
    Parameters
    ----------
    title: str
        Title of table
    headers: Union[list, tuple]
        header of table
    contents: Union[list, tuple]
        content of table
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
