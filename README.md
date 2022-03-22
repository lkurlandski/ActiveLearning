# ActiveLearning

An active learning experimental environment based upon open source active learning libraries and frameworks. Much of this is inspired by the NLP-ML-Research repository.

## Requirements

This codebase was written in Python 3.7.5. Third party requirements are detailed the requirements.txt file. 

## Setup

The following command-line arguments should be sufficient to set up users.

For ELSA users
- ssh {user-id}@elsa.hpc.tcnj.edu
- ssh dev1

Clone and enter repository
- git clone https://nlp-ml.hpc.tcnj.edu:8080/kurlanl1/ActiveLearning.git
- cd ActiveLearning

Set up a virtual environment with the correct version of Python
- module add python/3.7.5
- python3 -m venv env

Enter the virtual environment (do this every time you want to use this environment)
- source env/bin/activate

Install the required packages
- pip install -r requirements.txt

Install the active_learning/ source code as package
- pip install -e .

Users are advised to familiarize themselves with the engineering aspects of the Python language and execution evironment.

## Codebase Usage

All source code for this repository resides in the active_learning package. 

There are four scripts that perform the active learning experiments and analyze the data:
- active_learning/active_learner.py - runs active learning experiments
- active_learning/processor.py - processes raw output from active learning experiments into simple csv file formats
- active_learning/graphing.py - creates learning curves from the processed and stopping data
- active_learning/averager.py - averages experimental runs across several different random sampling versions

Each of these scripts can be run from the command line for a single set of experimental hyperparamerters. When run from the command line, they will use the experimental configurations specified in active_learning/local.py. To run from the command line,
- python active_learning/active_learner.py
- python active_learning/processor.py
- python active_learning/graphing.py
- python active_learning/averager.py

If we want to run all four of these scripts in succession, one after another, we can use active_learning/runner.py. 
- runner.py - interfaces to the active_learner, processor, stopping, graphing, and averager scripts

If we want to run several experiments, we use a active_learning/wrapper.py to run different combinations of experimental configurations. By default, this program will be executed on the cluster. The user can run this program locally by running the --local flag. By default, this will not run the averager.py program. The user can add the --averager flag to run the averager.py program instead of the active_learner.py, processor.py, and graphing.py programs. In summary, the possible ways to run wrapper.py are as follows:
- python active_learning/wrapper.py
- python active_learning/wrapper.py --local
- python active_learning/wrapper.py --average
- python active_learning/wrapper.py --local --average

## Contributing

Using a uniform coding style across all team members can improve the readability of the codebase and greatly improve diff tool performance.

- Use the highly readable numpy-style docstring (suggest using an IDE extension to automatically produce docstring)
- Use type hints in function arguments (suggest using an IDE extension to support type hinting)
- Limit the number of characters per line to 100 (suggest using an IDE extension to put vertical line at this position)

Before committing changes, use black's auto formatter to enforce a uniform coding style
- black --line-length 100 myfile.py

Pylint can identify errors in your code and provide other useful feedback. Consider addressing some of pylint's suggestions
- pylint myfile.py
