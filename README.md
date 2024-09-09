<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="https://www.ai.uni-hannover.de/typo3temp/focuscrop/e8a88c32efe940d6f6c2dbc8d64e3c6f49314de8-fp-16-9-0-0.jpg" alt="Logo" width="250px"/>
    <h1 align="center">Novelty-Guided Proximal Curriculum Learning</h1>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project
This project implements Novelty-Guided Proximal Curriculum Learning, which is based on Proximal Curriculum Learning [[1](#bibliography)] and Random Network Distillation [[2](#bibliography)]

<!-- GETTING STARTED -->
## Getting Started
### Installation
Download the project:

```shell
git clone https://github.com/CrunchyFlakes/Novelty-Guided-Proximal-Curriculum-Learning.git
cd Novelty-Guided-Proximal-Curriculum-Learning
```

In case you want the same versions of packages as were used by me, use `fixed_requirements.txt` instead of `requirements.txt` in the following:

#### Conda
Create a conda environment:

```bash
conda create -n ngpcl python=3.12
conda activate ngpcl
pip install -r requirements.txt
```

#### Venv
or a python venv:

```bash
# Make sure you have python 3.12
python -V
python -m venv ngpcl
./ngpcl/bin/activate
pip install -r requirements.txt
```

<!-- USAGE EXAMPLES -->
## Usage

You can run experiments by calling main.py. An example call which first does hpo on doorkey8 using novelty-guided proximal curriculum learning and then evaluates:

```shell
python main.py --approach_to_check comb --trials 100 --workers 20 --env_name doorkey --env_size 8 --mode hpo --n_seeds_hpo 1 --result_dir ./results/
python main.py --approach_to_check comb --env_name doorkey --env_size 8 --mode eval --n_seeds_eval 1 --result_dir ./results/ --eval_seed 0
```

All run results are included in the directories `results` and `plots`.
If you want to reproduce the results (due to SMAC parallelity results may differ slightly):

```shell
./full_run.sh 
```

## Results
Here is a short exempt out of the proposal. Read `proposal/proposal.pdf` for more information
<div align="center">
    <h4 align="center">Result plots for the different approaches</h4>
    <img src="plots/doorkey8/results_per_approach.svg" alt="cell structure" width="60%"/>
</div>

## Bibliography

[1]G. Tzannetos, B. G. Ribeiro, P. Kamalaruban, and A. Singla, “Proximal Curriculum for Reinforcement Learning Agents,” Trans. Mach. Learn. Res., vol. 2023, 2023, [Online]. Available: https://openreview.net/forum?id=8WUyeeMxMH

[2]Y. Burda, H. Edwards, A. J. Storkey, and O. Klimov, “Exploration by Random Network Distillation,” CoRR, vol. abs/1810.12894, 2018, [Online]. Available: http://arxiv.org/abs/1810.12894

[3]M. Lindauer et al., “SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization,” Journal of Machine Learning Research, vol. 23, no. 54, pp. 1–9, 2022, [Online]. Available: http://jmlr.org/papers/v23/21-0888.html
