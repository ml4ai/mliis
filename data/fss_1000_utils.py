
import glob
import os
import random
from typing import List


def split_train_test_tasks(all_tasks: List[str], n_test, reproducbile_splits: bool = False):
    if not isinstance(all_tasks, list):
        all_tasks = list(all_tasks)
    if reproducbile_splits:
        all_tasks = sorted(all_tasks)
    else:
        random.shuffle(all_tasks)
    test_set = []
    for i in range(n_test):
        test_set.append(all_tasks.pop())
    assert_train_test_split(all_tasks, test_set)
    return all_tasks, test_set


def assert_train_test_split(train, test):
    for i in test:
        assert i not in train, "train-test leakage"


def get_fss_tasks(data_dir):
    return glob.glob(os.path.join(data_dir, "*.tfrecord*"))


def get_fss_test_set() -> List[str]:
    dirname = os.path.dirname(__file__)
    path = "fss_test_set.txt"  # File containing the test examples from the FSS-1000 authors.
    filename = os.path.join(dirname, path)
    with open(filename, "r") as file:
        tasks = [line.rstrip("\n") for line in file]
    return tasks


def get_fss_train_set() -> List[str]:
    dirname = os.path.dirname(__file__)
    path = "fss_train_set.txt"
    filename = os.path.join(dirname, path)
    with open(filename, "r") as file:
        tasks = [line.rstrip("\n") for line in file]
    return tasks


def get_fp_k_test_set() -> List[str]:
    dirname = os.path.dirname(__file__)
    path = "fp-k_test_set.txt"  # File containing the test examples from the FSS-1000 authors.
    filename = os.path.join(dirname, path)
    with open(filename, "r") as file:
        tasks = [line.rstrip("\n") for line in file]
    return tasks


TEST_TASK_IDS = get_fss_test_set()
TRAIN_TASK_IDS = get_fss_train_set()
FP_K_TEST_TASK_IDS = get_fp_k_test_set()
TOTAL_NUM_FSS_CLASSES = 1000
IMAGE_DIMS = 224  # Length of one side of input images. Images assumed to be square.
