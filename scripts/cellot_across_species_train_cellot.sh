#! /bin/bash

# Phase 2. Train CellOT transport models after all Phase-1 scGen/AE jobs finish.
# AE holdout names are fixed as LPS6_${species}, matching condition_species.txt.

echo export OMP_NUM_THREADS=4
echo export OPENBLAS_NUM_THREADS=4
echo export MKL_NUM_THREADS=4
echo export VECLIB_MAXIMUM_THREADS=4
echo export NUMEXPR_NUM_THREADS=4
echo export NUMBA_NUM_THREADS=4

for seed in 1327 1337 1347; do
    log_dir="../Logs/across_species/CellOT/cellot/seed-${seed}"
    mkdir -p "${log_dir}"

    for species in $(cat ../Data/across_species/species.txt); do
        condition_species="LPS6_${species}"

        for mode in ood; do
            echo \
                nohup python cellot_train.py \
                --outdir ../cellot_results/across_species/seed-${seed}/holdout-${species}/mode-${mode}/model-cellot \
                --config ../cellot_configs/tasks/across_species.yaml \
                --config ../cellot_configs/models/cellot.yaml \
                --config.data.path ../Data/across_species/species.h5ad \
                --config.datasplit.holdout ${species} \
                --config.datasplit.key species \
                --config.datasplit.mode ${mode} \
                --config.datasplit.random_state 0 \
                --config.data.ae_emb.path ../cellot_results/across_species/seed-${seed}/holdout-${condition_species}/mode-${mode}/model-scgen \
                --seed ${seed} \
                --restart=True \
                \> "${log_dir}/cellot_holdout_${species}_seed${seed}.log" \
                2\>\&1 \
                \&;
        done
    done
 done
