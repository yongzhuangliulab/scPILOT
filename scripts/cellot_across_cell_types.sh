#! /bin/bash

for mode in ood; do
    for condition_cell_type in $(cat ../Data/across_cell_types/condition_cell_types.txt); do
        for model in scgen; do
            echo \
                python cellot_train.py \
                --outdir ../cellot_results/across_cell_types/holdout-${condition_cell_type}/mode-${mode}/model-${model} \
                --config ../cellot_configs/tasks/across_cell_types.yaml \
                --config ../cellot_configs/models/${model}.yaml \
                --config.data.path ../Data/across_cell_types/pbmc.h5ad \
                --config.datasplit.holdout $condition_cell_type \
                --config.datasplit.key condition_cell_type \
                --config.datasplit.mode $mode \
                --restart=True;
        done;
    done;
done;


# This is to use the same encoder as scgen
for mode in ood; do
    for cell_type in $(cat ../Data/across_cell_types/cell_types.txt); do
        echo \
            python cellot_train.py \
            --outdir ../cellot_results/across_cell_types/holdout-${cell_type}/mode-${mode}/model-cellot \
            --config ../cellot_configs/tasks/across_cell_types.yaml \
            --config ../cellot_configs/models/cellot.yaml \
            --config.data.path ../Data/across_cell_types/pbmc.h5ad \
            --config.datasplit.holdout $cell_type \
            --config.datasplit.key cell_type \
            --config.datasplit.mode $mode \
            --config.data.ae_emb.path ../cellot_results/across_cell_types/holdout-stimulated-${cell_type}/mode-${mode}/model-scgen \
            --restart=True;
    done;
done;