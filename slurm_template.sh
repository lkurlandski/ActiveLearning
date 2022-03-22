#!/home/hpc/kurlanl1/bloodgood/ActiveLearning/env/bin/python -u

#SBATCH --chdir=/home/hpc/kurlanl1/bloodgood/ActiveLearning
#SBATCH --job-name=20NewsGroups
#SBATCH --output=/home/hpc/kurlanl1/bloodgood/ActiveLearning/slurm/jobs/job.%A_%a.out
#SBATCH --signal=B:INT@600
#SBATCH --constraint="skylake|broadwell"
#SBATCH --partition=long



import os
import platform
import sys
import time
import atexit
import signal


from active_learning import runner

print('Running on:', platform.node())

# Create signal handler for SIGINT
def sigint_handler(signum, frame):
	print('INT signal handler called.  Exiting.')
	# Cleanup files
	print('Job interrupted at: ' + time.strftime('%m/%d/%Y %H:%M'))
	sys.exit(-1)

# Register SIGINT handler and atexit function (to ensure deletion of local files)
signal.signal(signal.SIGINT, sigint_handler)

# Start job and print out config file path
print('Starting runner.sh job for gc task at: ' + time.strftime('%m/%d/%Y %H:%M'))

runner.main()

# Job Finished
print('Job finished at: ' + time.strftime('%m/%d/%Y %H:%M'), flush=True)

