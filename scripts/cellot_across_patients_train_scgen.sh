#! /bin/bash

# Phase 1. Train one scGen/autoencoder model per held-out stimulated patient and seed.
# condition_sample_ids.txt is expected to contain entries such as stim_101.

echo export OMP_NUM_THREADS=4
echo export OPENBLAS_NUM_THREADS=4
echo export MKL_NUM_THREADS=4
echo export VECLIB_MAXIMUM_THREADS=4
echo export NUMEXPR_NUM_THREADS=4
echo export NUMBA_NUM_THREADS=4

for seed in 1327 1337 1347; do
    log_dir="../Logs/across_patients/CellOT/scgen/seed-${seed}"
    mkdir -p "${log_dir}"

    for condition_sample_id in $(cat ../Data/across_patients/condition_sample_ids.txt); do
        for mode in ood; do
            echo \
                nohup python cellot_train.py \
                --outdir ../cellot_results/across_patients/seed-${seed}/holdout-${condition_sample_id}/mode-${mode}/model-scgen \
                --config ../cellot_configs/tasks/across_patients.yaml \
                --config ../cellot_configs/models/scgen.yaml \
                --config.data.path ../Data/across_patients/pbmc_patients.h5ad \
                --config.datasplit.holdout ${condition_sample_id} \
                --config.datasplit.key condition_sample_id \
                --config.datasplit.mode ${mode} \
                --config.datasplit.random_state 0 \
                --seed ${seed} \
                --restart=True \
                \> "${log_dir}/scgen_holdout_${condition_sample_id}_seed${seed}.log" \
                2\>\&1 \
                \&;
        done
    done
done