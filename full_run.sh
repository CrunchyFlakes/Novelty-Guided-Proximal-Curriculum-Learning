for env in "unlock" "doorkey"; do
  for approach in "comb" "prox" "nov" "vanilla"; do
    python main.py --approach_to_check $approach --trials 100 --workers 20 --env_name $env --env_size 8 --mode hpo --n_seeds_hpo 1 --result_dir ./results/
    python main.py --approach_to_check $approach --env_name $env --env_size 8 --mode eval --n_seeds_eval 5 --result_dir ./results/ --eval_seed 0
  done
done
for context in "paper" "talk"; do
  python plotting.py --comb_result ./results/doorkey8_comb/result_info_seed0.json --nov_result ./results/doorkey8_nov/result_info_seed0.json --prox_result ./results/doorkey8_prox/result_info_seed0.json --vanilla_result ./results/doorkey8_vanilla/result_info_seed0.json --context $context --output_dir ./plots/doorkey8
  python plotting.py --comb_result ./results/unlock8_comb/result_info_seed0.json --nov_result ./results/unlock8_nov/result_info_seed0.json --prox_result ./results/unlock8_prox/result_info_seed0.json --vanilla_result ./results/unlock8_vanilla/result_info_seed0.json --context $context --output_dir ./plots/unlock8
done
