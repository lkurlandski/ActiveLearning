# Summary

An active learning experimental environment based upon open source active learning libraries and frameworks. Much of this is inspired by the NLP-ML-Research repository.

# Requirements

Python 3.7.5. Third party software as described in the requirements.txt file.

# Setup

The following command-line arguments should be sufficient to set up users.

For ELSA users
```console
ssh {user-id}@elsa.hpc.tcnj.edu
ssh dev1
```

Clone and enter repository
```console
git clone https://nlp-ml.hpc.tcnj.edu:8080/kurlanl1/ActiveLearning.git
cd ActiveLearning
```

Set up a virtual environment with the correct version of Python
```console
module load python/3.7.7
python3 -m venv env
```

Enter the virtual environment (do this every time you want to use this environment)
```console
source env/bin/activate
```

Install the required packages
```console
pip install --upgrade pip
pip install -r requirements.txt
```

Install the active_learning/ source code as package
```console
pip install -e .
```

Users are advised to familiarize themselves with the engineering aspects of the Python language and execution evironment.

# Usage

All source code for this repository resides in the active_learning package. The subpackages in the active_learning package are:
- al_pipeline
- dataset_fetchers
- feature_extractors

We will discuss the usage of these modules in a logical order: dataset_fetchers, feature_extractors, and finally al_pipeline.

## Fetching datasets with dataset_fetchers

TODO

## Extracting features with feature_extractors

TODO

## Running active learning experiments with al_pipeline

The al_pipeline package conducts active learning experiments and analyzes the results of the experiments. The modules in the al_pipeline package are
- average
- evaluate
- graph
- learn
- multirun
- output_helper
- process
- run

We will discuss the usage of these modules in a logical order: output_helper, learn, evaluate, process, graph, average, run and finally multirun.

### Managing output with the output_helper

The output_helper module assists with managing the experimental output paths.

### Query and teach with learn

The learn module performs the query, label, and teaching process of active learning. This is by far the most time consuming and computationally demanding portion of the entire package, so the object is to perform the minimal amount of computations needed to finish the process. All data processing and analysis is performed at a later time. To this end, certain essential files are saved throughout the learning process, such as the learned model at every iteration.

### Model inference and evaluation with evaluate

The evaluate module revives learned models (produced by the learn module) and evaluates their performance upon several pools data, such as a test set of entirely unseen data. This is another computationally demanding task, so again, the objective is to perform the minimal amount of computations needed to finish the process. At every iteration, the performance of the model is output to a new raw record and saved to disk. Creating new files is preferred to appending to an existing file because it is more computationally efficient.

### Processing raw data with process

The process module reads raw records (produced by the evaluate module) and processes them into neat, human-readable, and computer friendly tables.

### Creating graphs with graph

The graph module reads processed tables (produced by the process module) and creates basic, fundamental graphs to illuminate the learning process. The graphs are not meant to display complex trends; they are meant to be reliable, abundant, and readable. If users desire sophisticated graphs, they should write custom functions to produce them as needed.

### Averaging experimental results with average

The average module reads processed tables (produced by the process module) and averages them across several experimental runs. Not all data can be logically averaged between runs. The average module then requests that the graph module create graphs for the averaged experimental results.

### Running modules with run

The run module provides a simple and consistent interface the all of these modules (except the output_helper, which does not need an external interface).

The run module has a Python dict object named `params_`. The values in this dict control the experimental parameters used to perform the tasks provided by the run module. To use the run module, users need to edit this dict as needed.

When running locally, the run module has a command-line-interface with five boolean flags:
- --learn
- --evaluate
- --process
- --graph
- --average

Each flag refers to one of the modules with that flag's respective name. Providing the flag through the command line tells the run module to execute the respective module's processes using the experimental parameters in the `params_` dict. For example,

```console
python active_learning/al_pipeline/run.py --learn
```

will run only the learn module, while

```console
python active_learning/al_pipeline/run.py --evaluate
```

will run only the evaluate module. This second command will obviously throw an error if the correct files and directories do not exist to perform evaluation. In the fact, the first command is needed to proceed to second command for the second command to run properly. To avoid such confusions, both commands can be specified, like in


```console
python active_learning/al_pipeline/run.py --learn --evaluate
```

In general, the order does not matter, so the following is equivalent:

```console
python active_learning/al_pipeline/run.py --evaluate --learn
```

In fact, several processes can be specified at once, as in

```console
python active_learning/al_pipeline/run.py --learn --evaluate --process --graph
```

However, the average module should not be executed until all experiments it intends to average have completed. For this readon, an error is thrown if the user tries to combine the --average flag with any other flag(s). For example, the below will throw and error

```console
python active_learning/al_pipeline/run.py --learn --evaluate --process --graph --average
```

The correct way to do this would be to run
```console
python active_learning/al_pipeline/run.py --learn --evaluate --process --graph
```

and wait for all experiments to finish. Then run

```console
python active_learning/al_pipeline/run.py --average
```

Finally, for instructions on how to use the run module from the command line, simply type
```console
python active_learning/al_pipeline/run.py -h
```

The run module is also executable as a SLURM script. Before executing as a SLURM script, first modify the lines at the top of the script to point to the correct locations. The interface as a slurm script is the exact same as the interface as a python script. Simply use the sbatch command instead of the python command. For example,
```console
sbatch active_learning/al_pipeline/run.py --learn --evaluate --process --graph
```

Using SLURM comes with a host of benefits. First of all, it allows for multiprocessing. When performing one-vs-rest classification with n_classes, n_classes classifiers are trained and ensembled to create a meta-classifier. This process can be efficiently parallelized by using n_classes threads. The codebase is already configured to perform multithreading if possible. However, by default, SLURM only allocates one cpu to a submitted job, which diminishes the potential for multithreading. We can ammend this situtation with SLURM's --cpus-per-task parameter. For example, suppose we are running an experiment with 20NewsGroups and support vector machines in a one-vs-rest configuration. Our submission to sbatch could look like
```console
sbatch active_learning/al_pipeline/run.py --learn --cpus-per-task=20
```

For more details on possible options to use with sbatch, see 
```console
sbatch -h
```

Please note that configurations specified at the top of the run script will overwrite any configurations passed through the command line. For example, if the top of run.py contains
```
#SBATCH --partition=short
```

but you submit the script to SLURM using
```console
sbatch active_learning/al_pipeline/run.py --learn --partition=long
```

your job will be submitted with the short partition, not the long one!

### Running several experiments with multirun

To run several experiments with a single command, we can use the mutltirun script. This script has less flexibiltiy than run, but can be useful nonetheless. This script operates by creating executable files and submitting them with sbatch inside of a looping structure. 

To use the multirun program, first, edit the `root_path` variable to point to the location of the ActiveLearning directory.. Next, edit the `multi_params_` dict inside of it with the different experiments you would like to run. Then edit the `slurm_precepts` with any custom sbatch options you would like the job to be submited with. Then proceed with the command-line interface described for the run script. For example,

```console
python active_learning/al_pipeline/multirun.py --learn --evaluate --process --graph
```

# Contributing

Using a uniform coding style across all team members can improve the readability of the codebase and greatly improve diff tool performance.

- Use the highly readable numpy-style docstring (suggest using an IDE extension to automatically produce docstring)
- Use type hints in function arguments (suggest using an IDE extension to support type hinting)
- Limit the number of characters per line to 100 (suggest using an IDE extension to put vertical line at this position)

Before committing changes, use black's auto formatter to enforce a uniform coding style
- black --line-length 100 --exclude active_learning/al_pipeline/run.py active_learning

Pylint can identify errors in your code and provide other useful feedback. Consider addressing some of pylint's suggestions
- pylint myfile.py

Run the unit tests to make sure you did not break anything. Edit and add new unit tests if required.
- pytest tests
