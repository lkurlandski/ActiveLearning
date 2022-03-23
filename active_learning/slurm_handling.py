import shutil
import signal, os, sys

node_root = "/local/scratch/"
interrupted = False
job_id_list = []
user_path = ""

def move_output():
    for job_id in job_id_list:
        shutil.move(node_root + job_id, user_path)

def signal_handler(sigint, frame):
    global interrupted
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)

if interrupted:
    print("CAUGHT SIGNAL")