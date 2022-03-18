"""Administration tasks for the team admin."""

from pprint import pformat
import subprocess

def get_our_nodes(cpu=True, gpu=True):

    nodes = []

    if cpu:
        nodes += [f"node06{i}" for i in range(1, 10)]
        nodes += [f"node07{i}" for i in range(0, 10)]
        nodes += [f"node08{i}" for i in range(0, 2)]
        nodes += [f"node13{i}" for i in range(1, 9)]

    if gpu:
        nodes += [f"gpu-node00{i}" for i in range(1, 10)]
        nodes += [f"gpu-node01{i}" for i in range(0, 19)]

    return nodes

def analyze_disk_space(nodes):

    for n in nodes:
        report = subprocess.run(["ssh", n, "df", "-h", "/local/scratch"], check=True, stdout=subprocess.PIPE)
        report = str(report.stdout.decode("utf-8"))
        print(f"Node {n}:\n{report}")

def detect_files(nodes):

    for n in nodes:
        dirs = subprocess.run(["ssh", n, "ls", "/local/scratch"], check=True, stdout=subprocess.PIPE)
        dirs = str(dirs.stdout.decode("utf-8")).split()
        print(f"Node {n}:\n{pformat(sorted(dirs))}\n-------------------------\n")
