#!/bin/bash

echo export OMP_NUM_THREADS=4
echo export OPENBLAS_NUM_THREADS=4
echo export MKL_NUM_THREADS=4
echo export VECLIB_MAXIMUM_THREADS=4
echo export NUMEXPR_NUM_THREADS=4
echo export NUMBA_NUM_THREADS=4

for seed in 1327 1337 1347; do
    log_dir="../Logs/across_cell_types/scPILOT/seed-${seed}"
    mkdir -p "${log_dir}"

    for query_key in $(cat ../Data/across_cell_types/cell_labels.txt); do
        echo \
            nohup python across_cell_types_perturbation_prediction.py \
            --query_key ${query_key} \
            --seed ${seed} \
            --split_seed 0 \
            \> "${log_dir}/across_cell_types_scPILOT_${query_key}_seed${seed}.log" \
            2\>\&1 \
            \&;
    done;
done;
