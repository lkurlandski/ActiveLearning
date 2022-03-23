import shutil
import signal, os, sys

node_root = "/local/scratch/"
interrupted = False
job_id_list = []
user_path = ""

def move_output():
    print("moving the output")
    for job_id in job_id_list:
        print(job_id)
        shutil.move(node_root + job_id, user_path)