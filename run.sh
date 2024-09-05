#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH -J doorkey8
#SBATCH --mem=32G
#SBATCH -o slurm_outputs/slurm-%j.out
#SBATCH --partition=ai
#SBATCH --time=12:00:00

#SBATCH --mail-user=m.toepperwien@stud.uni-hannover.de
#SBATCH --mail-type=FAIL,END
cd $SLURM_SUBMIT_DIR

module load Miniconda3

source /home/nhwptopm/.bashrc

conda activate /bigwork/nhwptopm/.conda/envs/adrl

export WANDB_MODE=offline

python /bigwork/nhwptopm/adrl_proximal_curriculum_w_novelty/main.py --workers=20 --trials=100 --env_name doorkey --env_size 8 --n_seeds_train 1 --n_seeds_eval 1 --smac_output_dir smac3_output/${SLURM_JOB_ID}_${SLURM_JOB_NAME}
