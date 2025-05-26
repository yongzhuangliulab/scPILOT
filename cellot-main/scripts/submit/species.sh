#! /bin/bash

for mode in ood iid; do
    for condition_species in $(cat datasets/species/condition_species.txt); do
        for model in scgen; do
            echo \
                python ./scripts/train.py \
                --outdir ./results/species/holdout-${condition_species}/mode-${mode}/model-${model} \
                --config ./configs/tasks/species.yaml \
                --config ./configs/models/${model}.yaml \
                --config.data.path ./datasets/species/species.h5ad \
                --config.datasplit.holdout $condition_species \
                --config.datasplit.key condition_species \
                --config.datasplit.mode $mode \
                --restart=True;
        done;
    done;
done;


# This is to use the same encoder as scgen
for mode in ood iid; do
    for species in $(cat datasets/species/species.txt); do
        echo \
            python ./scripts/train.py \
            --outdir ./results/species/holdout-${species}/mode-${mode}/model-cellot \
            --config ./configs/tasks/species.yaml \
            --config ./configs/models/cellot.yaml \
            --config.data.path ./datasets/species/species.h5ad \
            --config.datasplit.holdout LPS6_${species},unst_${species} \
            --config.datasplit.key condition_species \
            --config.datasplit.mode $mode \
            --config.data.ae_emb.path ./results/species/holdout-LPS6_${species}/mode-${mode}/model-scgen \
            --restart=True;
    done;
done;