# -*- coding: UTF-8 -*-
# RFDFM source code
# May-22-2024
import os
import shutil
from datetime import datetime

from about_log import logger


def delete_dir_if_exists(directory, directly=False, warning=False):
    """
    Delete directory if exists, If `directly`, delete without message prompt otherwise user should confirm.

    :param directory: Where to delete recursively.
    :param directly: Whether to delete directly.
    :param warning: WARNING INFO if ture
    :return:
    """
    if os.path.exists(directory):
        if not directly:
            logger.warning(f"{directory} already exists! Delete it? yes[y]/No[n]")
            i = input()
            if i.lower() == 'y' or i.lower() == 'yes':
                directly = True
        if directly:
            shutil.rmtree(directory, ignore_errors=True)
            logger.info(f"Successfully delete directory {directory}")
    elif warning:
        logger.warning(f"{directory} not exists!")


def create_directories_if_not_exists(*directories, truncate=False):
    """Create directories if not exists.

    :param directories: Directory to create.
    :param truncate: Truncate directory or not.
    """
    for directory in directories:
        if truncate:
            shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory, exist_ok=True)


def create_dir_if_not_exists(directory, add_date=False, add_time=False):
    """Create directory if not exists.

    :param directory: Directory to create.
    :param add_date: Add date directory. If True, `directory/DATE` will be created.
    :param add_time: Add datetime directory. If True, `directory/DATETIME` will be created.
    :return path: The created path.
    """
    path = None
    if directory:
        path = directory
        if add_date:
            path = os.path.join(path, datetime.now().strftime("%Y%m%d"))
        if add_time:
            path = os.path.join(path, datetime.now().strftime("%H%M%S"))
        os.makedirs(path, exist_ok=True)
    return path


def truncate_dir(directory, del_directly=False, **kwargs):
    """
    Truncate directory.

    :param directory: Which directory to be truncated!
    :param del_directly:  Delete directory directly or not.
    :return:
    """
    delete_dir_if_exists(directory, del_directly)
    return create_dir_if_not_exists(directory, **kwargs)


def check_directory_exists(*directories, prefixes=''):
    """Check directory if not exists raise IOError."""
    # Check prefix's length and type.
    if type(prefixes) == list or type(prefixes) == tuple:
        assert len(prefixes) == len(directories), 'length of `prefix` and `directories` must be equal!'
    elif type(prefixes) == str:
        prefixes = [prefixes] * len(directories)
    for directory, prefix in zip(directories, prefixes):
        if directory and not os.path.exists(directory):
            raise IOError('%s: %s not exists!' % (prefix, directory))
