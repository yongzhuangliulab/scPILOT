#! /bin/bash

for mode in ood iid; do
    for condition_population in $(cat datasets/differentiation/condition_populations.txt); do
        for model in scgen; do
            echo \
                python ./scripts/train.py \
                --outdir ./results/differentiation/holdout-${condition_population}/mode-${mode}/model-${model} \
                --config ./configs/tasks/differentiation.yaml \
                --config ./configs/models/${model}.yaml \
                --config.data.path ./datasets/differentiation/differentiation.h5ad \
                --config.datasplit.holdout $condition_population \
                --config.datasplit.key condition_population \
                --config.datasplit.mode $mode \
                --restart=True;
        done;
    done;
done;


# This is to use the same encoder as scgen
for mode in ood iid; do
    for population in $(cat datasets/differentiation/populations.txt); do
        echo \
            python ./scripts/train.py \
            --outdir ./results/differentiation/holdout-${population}/mode-${mode}/model-cellot \
            --config ./configs/tasks/differentiation.yaml \
            --config ./configs/models/cellot.yaml \
            --config.data.path ./datasets/differentiation/differentiation.h5ad \
            --config.datasplit.holdout developed_${population},control_${population} \
            --config.datasplit.key condition_population \
            --config.datasplit.mode $mode \
            --config.data.ae_emb.path ./results/differentiation/holdout-developed_${population}/mode-${mode}/model-scgen \
            --restart=True;
    done;
done;