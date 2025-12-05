#! /bin/bash

for mode in ood; do
    for condition_species in $(cat ../Data/across_species/condition_species.txt); do
        for model in scgen; do
            echo \
                python cellot_train.py \
                --outdir ../cellot_results/across_species/holdout-${condition_species}/mode-${mode}/model-${model} \
                --config ../cellot_configs/tasks/across_species.yaml \
                --config ../cellot_configs/models/${model}.yaml \
                --config.data.path ../Data/across_species/species.h5ad \
                --config.datasplit.holdout $condition_species \
                --config.datasplit.key condition_species \
                --config.datasplit.mode $mode \
                --restart=True;
        done;
    done;
done;


# This is to use the same encoder as scgen
for mode in ood; do
    for species in $(cat ../Data/across_species/species.txt); do
        echo \
            python cellot_train.py \
            --outdir ../cellot_results/across_species/holdout-${species}/mode-${mode}/model-cellot \
            --config ../cellot_configs/tasks/across_species.yaml \
            --config ../cellot_configs/models/cellot.yaml \
            --config.data.path ../Data/across_species/species.h5ad \
            --config.datasplit.holdout $species \
            --config.datasplit.key species \
            --config.datasplit.mode $mode \
            --config.data.ae_emb.path ../cellot_results/across_species/holdout-LPS6_${species}/mode-${mode}/model-scgen \
            --restart=True;
    done;
done;