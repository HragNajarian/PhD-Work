# Figure out how many number of arrays you need when you sbatch .job
	# After running this, run: sbatch --array=1-$N run_swap_vars_met_em.job

source config.sh
source utils.sh

N=$(generate_hours $START1 $END1 | wc -w)
echo "Number of array jobs: $N"
