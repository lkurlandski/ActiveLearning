
from pathlib import Path
import shutil
from glob import glob
import os


user_path = ""
node_root = "/local/scratch/bloodgood/"


def move_output():
    subdirs1 = Path(node_root).glob("**")
    subdirs2 = glob("/local/scratch/bloodgood/*/")
    print(subdirs1)
    print(subdirs2)

    for dir in subdirs1:
        print(dir)
        #if not(Path.exists(dir)):
            #os.mkdir(dir)
        shutil.copy(dir, user_path)