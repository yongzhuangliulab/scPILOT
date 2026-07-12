#! /bin/bash

# Phase 1. Train one scGen/autoencoder model per held-out stimulated species and seed.
# condition_species.txt is expected to contain entries such as LPS6_mouse.

echo export OMP_NUM_THREADS=4
echo export OPENBLAS_NUM_THREADS=4
echo export MKL_NUM_THREADS=4
echo export VECLIB_MAXIMUM_THREADS=4
echo export NUMEXPR_NUM_THREADS=4
echo export NUMBA_NUM_THREADS=4

for seed in 1327 1337 1347; do
    log_dir="../Logs/across_species/CellOT/scgen/seed-${seed}"
    mkdir -p "${log_dir}"

    for condition_species in $(cat ../Data/across_species/condition_species.txt); do
        for mode in ood; do
            echo \
                nohup python cellot_train.py \
                --outdir ../cellot_results/across_species/seed-${seed}/holdout-${condition_species}/mode-${mode}/model-scgen \
                --config ../cellot_configs/tasks/across_species.yaml \
                --config ../cellot_configs/models/scgen.yaml \
                --config.data.path ../Data/across_species/species.h5ad \
                --config.datasplit.holdout ${condition_species} \
                --config.datasplit.key condition_species \
                --config.datasplit.mode ${mode} \
                --config.datasplit.random_state 0 \
                --seed ${seed} \
                --restart=True \
                \> "${log_dir}/scgen_holdout_${condition_species}_seed${seed}.log" \
                2\>\&1 \
                \&;
        done
    done
done
