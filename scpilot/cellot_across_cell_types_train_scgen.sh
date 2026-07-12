#! /bin/bash

# Phase 1. Train one scGen/autoencoder model per held-out stimulated cell type and seed.
# condition_cell_types.txt is expected to contain entries such as stimulated_B.

echo export OMP_NUM_THREADS=4
echo export OPENBLAS_NUM_THREADS=4
echo export MKL_NUM_THREADS=4
echo export VECLIB_MAXIMUM_THREADS=4
echo export NUMEXPR_NUM_THREADS=4
echo export NUMBA_NUM_THREADS=4

for seed in 1327 1337 1347; do
    log_dir="../Logs/across_cell_types/CellOT/scgen/seed-${seed}"
    mkdir -p "${log_dir}"

    for condition_cell_type in $(cat ../Data/across_cell_types/condition_cell_types.txt); do
        for mode in ood; do
            echo \
                nohup python cellot_train.py \
                --outdir ../cellot_results/across_cell_types/seed-${seed}/holdout-${condition_cell_type}/mode-${mode}/model-scgen \
                --config ../cellot_configs/tasks/across_cell_types.yaml \
                --config ../cellot_configs/models/scgen.yaml \
                --config.data.path ../Data/across_cell_types/pbmc.h5ad \
                --config.datasplit.holdout ${condition_cell_type} \
                --config.datasplit.key condition_cell_type \
                --config.datasplit.mode ${mode} \
                --config.datasplit.random_state 0 \
                --seed ${seed} \
                --restart=True \
                \> "${log_dir}/scgen_holdout_${condition_cell_type}_seed${seed}.log" \
                2\>\&1 \
                \&;
        done
    done
done
