#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
from utils import logger
import os

from .classification import arguments_classification, build_classification_model


SUPPORTED_TASKS = [
    name.lower()
    for name in os.listdir(".")
    if os.path.isdir(name)
    and name.find("__") == -1  # check if it is a directory and not a __pycache__ folder
]


def arguments_model(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # classification network
    parser = arguments_classification(parser=parser)

    return parser


def get_model(opts):
    dataset_category = getattr(opts, "dataset.category", "classification")
    if dataset_category is None:
        task_str = "--dataset.category cannot be None. Supported categories are:"
        for i, task_name in enumerate(SUPPORTED_TASKS):
            task_str += "\n\t {}: {}".format(i, task_name)
        logger.error(task_str)

    dataset_category = dataset_category.lower()

    model = None
    if dataset_category == "classification":
        model = build_classification_model(opts=opts)
    else:
        task_str = (
            "Got {} as a task. Unfortunately, we do not support it yet."
            "\nSupported tasks are:".format(dataset_category)
        )
        for i, task_name in enumerate(SUPPORTED_TASKS):
            task_str += "\n\t {}: {}".format(i, task_name)
        logger.error(task_str)

    return model
