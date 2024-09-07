#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH -J unlock_comb
#SBATCH --mem=32G
#SBATCH -o slurm_outputs/slurm-%j.out
#SBATCH --partition=ai
#SBATCH --time=16:00:00

#SBATCH --mail-user=m.toepperwien@stud.uni-hannover.de
#SBATCH --mail-type=FAIL,END
cd $SLURM_SUBMIT_DIR

module load Miniconda3

source /home/nhwptopm/.bashrc

conda activate /bigwork/nhwptopm/.conda/envs/adrl

export WANDB_MODE=offline


python /bigwork/nhwptopm/adrl_proximal_curriculum_w_novelty/main.py --approach_to_check comb --trials 100 --workers 20 --env_name doorkey --env_size 8 --mode hpo --n_seeds_hpo 1 --result_dir ./results/${SLURM_JOB_ID}_${SLURM_JOB_NAME}
python main.py --approach_to_check comb --env_name doorkey --env_size 8 --mode eval --n_seeds_eval 5 --result_dir ./results/${SLURM_JOB_ID}_${SLURM_JOB_NAME} --eval_seed 0
