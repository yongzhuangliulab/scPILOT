#! /bin/bash

for mode in ood; do
    for condition_sample_id in $(cat datasets/pbmc_patients/condition_sample_ids.txt); do
        for model in scgen; do
            echo \
                python ./scripts/train.py \
                --outdir ./results/pbmc_patients/holdout-${condition_sample_id}/mode-${mode}/model-${model} \
                --config ./configs/tasks/pbmc_patients.yaml \
                --config ./configs/models/${model}.yaml \
                --config.data.path ./datasets/pbmc_patients/pbmc_patients.h5ad \
                --config.datasplit.holdout $condition_sample_id \
                --config.datasplit.key condition_sample_id \
                --config.datasplit.mode $mode \
                --restart=True;
        done;
    done;
done;


# This is to use the same encoder as scgen
for mode in ood; do
    for sample_id in $(cat datasets/pbmc_patients/sample_ids.txt); do
        echo \
            python ./scripts/train.py \
            --outdir ./results/pbmc_patients/holdout-${sample_id}/mode-${mode}/model-cellot \
            --config ./configs/tasks/pbmc_patients.yaml \
            --config ./configs/models/cellot.yaml \
            --config.data.path ./datasets/pbmc_patients/pbmc_patients.h5ad \
            --config.datasplit.holdout $sample_id \
            --config.datasplit.key sample_id \
            --config.datasplit.mode $mode \
            --config.data.ae_emb.path ./results/pbmc_patients/holdout-stim-${sample_id}/mode-${mode}/model-scgen \
            --restart=True;
    done;
done;