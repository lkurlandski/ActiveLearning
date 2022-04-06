import shutil
import signal, os, sys

node_root = "/local/scratch/"
interrupted = False
job_id_list = []
user_path = ""

def move_output(experiment_parameters):
    #need to check if the directory exists before trying to move
    print("Moving output from node file system to user file system.")
    for task in experiment_parameters["task"]:
        shutil.move(node_root, user_path)