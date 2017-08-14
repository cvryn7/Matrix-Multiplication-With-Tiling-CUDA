#!/bin/bash -l
# NOTE the -l flag!
#

# This is an example job file for a multi-core job.
# Note that all of the following statements below that begin
# with #SBATCH are actually commands to the SLURM scheduler.
# Please copy this file to your home directory and modify it
# to suit your needs.
# 
# If you need any help, please email rc-help@rit.edu
#

# Name of the job - You'll probably want to customize this.
#SBATCH -J matmul_test_tile

# Standard out and Standard Error output files
#SBATCH -o matmul_test_tile.output
#SBATCH -e matmul_test_error_tile.output

#To send emails, set the adcdress below and remove one of the "#" signs.
##SBATCH --mail-user xyz@rit.edu

# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL

# Request 5 hours run time MAX, anything over will be KILLED
#SBATCH -t 5:0:0

# Put the job in the "work" partition and number of tasks as 1
# "work" is the default partition so it can be omitted without issue.
#SBATCH -p work -n 1

# Job memory requirements in MB
#SBATCH --mem=300

# Explicitly state you are a free user
#SBATCH --qos=free

#
# Your job script goes below this line.  
#

# Compile open mp C program
module load cuda

nvcc matrix_mul_tiling.cu -lcudart -o matmultile

# Run the omp executable
srun matmultile in_a.txt in_b.txt 4 5 4
