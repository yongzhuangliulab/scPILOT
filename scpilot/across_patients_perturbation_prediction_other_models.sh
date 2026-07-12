#!/bin/bash

echo export OMP_NUM_THREADS=4
echo export OPENBLAS_NUM_THREADS=4
echo export MKL_NUM_THREADS=4
echo export VECLIB_MAXIMUM_THREADS=4
echo export NUMEXPR_NUM_THREADS=4
echo export NUMBA_NUM_THREADS=4

for seed in 1327 1337 1347; do
    for model_name in biolord identity scGen; do
        log_dir="../Logs/across_patients/${model_name}/seed-${seed}"
        mkdir -p "${log_dir}"

        for query_key in $(cat ../Data/across_patients/sample_ids.txt); do
            echo \
                nohup python across_patients_perturbation_prediction_other_models.py \
                --model_name ${model_name} \
                --query_key ${query_key} \
                --seed ${seed} \
                --split_seed 0 \
                \> "${log_dir}/across_patients_${model_name}_${query_key}_seed${seed}.log" \
                2\>\&1 \
                \&;
        done;
    done;
done;