#! /bin/bash

for mode in ood; do
    for condition_cell_type in $(cat datasets/pbmc/condition_cell_types.txt); do
        for model in scgen; do
            echo \
                python ./scripts/train.py \
                --outdir ./results/pbmc/holdout-${condition_cell_type}/mode-${mode}/model-${model} \
                --config ./configs/tasks/pbmc.yaml \
                --config ./configs/models/${model}.yaml \
                --config.data.path ./datasets/pbmc/pbmc.h5ad \
                --config.datasplit.holdout $condition_cell_type \
                --config.datasplit.key condition_cell_type \
                --config.datasplit.mode $mode \
                --restart=True;
        done;
    done;
done;


# This is to use the same encoder as scgen
for mode in ood; do
    for cell_type in $(cat datasets/pbmc/cell_types.txt); do
        echo \
            python ./scripts/train.py \
            --outdir ./results/pbmc/holdout-${cell_type}/mode-${mode}/model-cellot \
            --config ./configs/tasks/pbmc.yaml \
            --config ./configs/models/cellot.yaml \
            --config.data.path ./datasets/pbmc/pbmc.h5ad \
            --config.datasplit.holdout $cell_type \
            --config.datasplit.key cell_type \
            --config.datasplit.mode $mode \
            --config.data.ae_emb.path ./results/pbmc/holdout-stimulated-${cell_type}/mode-${mode}/model-scgen \
            --restart=True;
    done;
done;