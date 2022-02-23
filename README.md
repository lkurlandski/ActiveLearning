# ActiveLearning

An active learning experimental environment based upon open source active learning libraries and frameworks. Much of this is inspired by the NLP-ML-Research repository, but is designed to be more modular, light weight, and simpler.

## Technical Requirements

Codebase is written in Python 3.7.5. Goal is to upgrade this to Python 3.9.9. Requirements are contained in requirements.txt. 

Currently, the system uses a slurm_template.sh file to create slurm scripts and execute them. The first line is this file tells the slurm script where to find the Python interpreter to use. The first line is '#!./env/bin/python -u', which indicates that it is looking for a Python interpreter in the ./env/bin/python location. Eventually, we will want to improve upon this clumsy behavior, but for now, you can simply perform the following:

> python3 -m venv env

> pip install -r requirements.txt

> source env/bin/activate

Users are advised to familiarize themselves with the engineering aspects of the Python language and execution evironment.

## Codebase Documentation

The main components of this system are:
- active_learner.py - runs active learning experiments
- processor.py - processes raw output from active learning experiments into simple csv file formats
- stopping.py - runs rudimentary stopping algorithms to compute where Stabilizing Predictions stops
- graphing.py - creates learning curves from the processed and stopping data
- averager.py - averages experimental runs across several different random sampling versions

Each of these scripts can be run from the command line by changing the experiment_parameters dictionary to perform the process for a single set of active learning hyperparamerters. 

To run a many active learning experiments with several different options for the various parameters on the HPC, you need to use the following, which can all be easily controlled from wrapper.py:
- driver.py - creates configuration files, creates slurm scripts, calls sbatch to run slurm scripts
- main.py - interfaces to the active_learner, processor, stopping, graphing, and averager scripts
- wrapper.py - interfaces to the driver and the main scripts

Additional assets include:
- config.py - system-level configurations
- estimators.py - handle base learners and estimators
- input_helper.py - handle dataset input
- output_helper.py - manage the complex output structure of active learning experiments
- state_helper.py - contains useful statistical functions
- stopping_methods.py - (prototype) assets for a future iteration of stopping criterion research
- vectorizers.py - (prototype) assets to handle processing raw text into numerical format
- verify.py - (depreciated) assets to check if all files are present in the output path
