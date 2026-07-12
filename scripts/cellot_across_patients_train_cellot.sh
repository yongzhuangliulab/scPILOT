#! /bin/bash

# Phase 2. Train CellOT transport models after all Phase-1 scGen/AE jobs finish.
# AE holdout names are fixed as stim_${sample_id}, matching condition_sample_ids.txt.

echo export OMP_NUM_THREADS=4
echo export OPENBLAS_NUM_THREADS=4
echo export MKL_NUM_THREADS=4
echo export VECLIB_MAXIMUM_THREADS=4
echo export NUMEXPR_NUM_THREADS=4
echo export NUMBA_NUM_THREADS=4

for seed in 1327 1337 1347; do
    log_dir="../Logs/across_patients/CellOT/cellot/seed-${seed}"
    mkdir -p "${log_dir}"

    for sample_id in $(cat ../Data/across_patients/sample_ids.txt); do
        condition_sample_id="stim_${sample_id}"

        for mode in ood; do
            echo \
                nohup python cellot_train.py \
                --outdir ../cellot_results/across_patients/seed-${seed}/holdout-${sample_id}/mode-${mode}/model-cellot \
                --config ../cellot_configs/tasks/across_patients.yaml \
                --config ../cellot_configs/models/cellot.yaml \
                --config.data.path ../Data/across_patients/pbmc_patients.h5ad \
                --config.datasplit.holdout ${sample_id} \
                --config.datasplit.key sample_id \
                --config.datasplit.mode ${mode} \
                --config.datasplit.random_state 0 \
                --config.data.ae_emb.path ../cellot_results/across_patients/seed-${seed}/holdout-${condition_sample_id}/mode-${mode}/model-scgen \
                --seed ${seed} \
                --restart=True \
                \> "${log_dir}/cellot_holdout_${sample_id}_seed${seed}.log" \
                2\>\&1 \
                \&;
        done
    done
done