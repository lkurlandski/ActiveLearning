
from pathlib import Path, PosixPath
import shutil
from glob import glob
import os

from active_learning import local


user_path = ""
node_root = "/local/scratch/bloodgood/"


def move_output():
    all_subdirs_pp = list(Path(node_root).glob("**"))
    all_subdirs_str = []
    subdirs = []

    #convert PosixPath to strings
    for subdir in all_subdirs_pp:
        all_subdirs_str.append(str(subdir))

    #get all the subdirectories with the dataset
    for subdir in all_subdirs_str:
        if local.experiment_parameters["dataset"] in subdir:
            subdirs.append(subdir)

    subdir = subdirs[0]
    subdir_local = user_path + subdir.replace("/local/scratch/bloodgood", "")

    #if the root subdirectory with the dataset exists, delete it
    if Path(subdir_local).exists():
        shutil.rmtree(subdir_local)

    #copy the contents to user path
    shutil.copytree(subdir, subdir_local)

    #remove the files from /local/scratch/bloodgood
    #if more tasks are added, loop through and delete each one
    shutil.rmtree(node_root + local.experiment_parameters["task"])
