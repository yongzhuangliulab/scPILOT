#! /bin/bash

# Phase 2. Train CellOT transport models after all Phase-1 scGen/AE jobs finish.
# AE holdout names are fixed as IFNGR2_${cell_type}, matching target_gene_cell_types.txt.

echo export OMP_NUM_THREADS=4
echo export OPENBLAS_NUM_THREADS=4
echo export MKL_NUM_THREADS=4
echo export VECLIB_MAXIMUM_THREADS=4
echo export NUMEXPR_NUM_THREADS=4
echo export NUMBA_NUM_THREADS=4

for seed in 1327 1337 1347; do
    log_dir="../Logs/across_cell_lines/CellOT/cellot/seed-${seed}"
    mkdir -p "${log_dir}"

    for cell_type in $(cat ../Data/across_cell_lines/cell_types.txt); do
        target_gene_cell_type="IFNGR2_${cell_type}"

        for mode in ood; do
            echo \
                nohup python cellot_train.py \
                --outdir ../cellot_results/across_cell_lines/seed-${seed}/holdout-${cell_type}/mode-${mode}/model-cellot \
                --config ../cellot_configs/tasks/across_cell_lines.yaml \
                --config ../cellot_configs/models/cellot.yaml \
                --config.data.path ../Data/across_cell_lines/IFNGR2.h5ad \
                --config.datasplit.holdout ${cell_type} \
                --config.datasplit.key cell_type \
                --config.datasplit.mode ${mode} \
                --config.datasplit.random_state 0 \
                --config.data.ae_emb.path ../cellot_results/across_cell_lines/seed-${seed}/holdout-${target_gene_cell_type}/mode-${mode}/model-scgen \
                --seed ${seed} \
                --restart=True \
                \> "${log_dir}/cellot_holdout_${cell_type}_seed${seed}.log" \
                2\>\&1 \
                \&;
        done
    done
done
