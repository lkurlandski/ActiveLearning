"""Administration tasks for the team admin.

TODO
----
-

FIXME
-----
-
"""

from pprint import pformat
import subprocess
from typing import List


def get_our_nodes(cpu: bool = True, gpu: bool = True) -> List[str]:
    """Return the nodes we commonly use.

    Parameters
    ----------
    cpu : bool, optional
        If True, includes the cpu nodes (broadwell and skylake), by default True.
    gpu : bool, optional
        If True, includes the gpu nodes (?), by default True.

    Returns
    -------
    List[str]
        List of the node names we use.
    """

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


def analyze_disk_space(nodes: List[str]) -> None:
    """Analyze the disk space of the /local/scratch directory on an ELSA hpc node.

    Parameters
    ----------
    nodes : List[str]
        List of nodes to analyze.
    """

    for n in nodes:
        report = subprocess.run(
            ["ssh", n, "df", "-h", "/local/scratch"], check=True, stdout=subprocess.PIPE
        )
        report = str(report.stdout.decode("utf-8"))
        print(f"Node {n}:\n{report}")


def detect_files(nodes: List[str]) -> None:
    """Detect potentially stale/wasteful files in the /local/acratch directory of ELSA's hpc nodes.

    Parameters
    ----------
    nodes : List[str]
        List of nodes to analyze.
    """

    for n in nodes:
        dirs = subprocess.run(
            ["ssh", n, "ls", "/local/scratch"], check=True, stdout=subprocess.PIPE
        )
        dirs = str(dirs.stdout.decode("utf-8")).split()
        print(f"Node {n}:\n{pformat(sorted(dirs))}\n-------------------------\n")
