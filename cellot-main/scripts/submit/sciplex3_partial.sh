#! /bin/bash

for drug in $(cat datasets/sciplex3_partial/drugs.txt); do
    for mode in ood iid; do
        for cell_type in $(cat datasets/sciplex3_partial/cell_types.txt); do
            for model in scgen; do
                echo \
                    python ./scripts/train.py \
                    --outdir ./results/sciplex3_partial/holdout-${drug}_${cell_type}/mode-${mode}/model-${model} \
                    --config ./configs/tasks/sciplex3_partial.yaml \
                    --config ./configs/models/${model}.yaml \
                    --config.data.path ./datasets/sciplex3_partial/sciplex3_partial.h5ad \
                    --config.data.target $drug \
                    --config.datasplit.holdout ${drug}_${cell_type} \
                    --config.datasplit.key drug_cell_type \
                    --config.datasplit.mode $mode \
                    --restart=True;
            done;
        done;
    done;
done;


# This is to use the same encoder as scgen
for drug in $(cat datasets/sciplex3_partial/drugs.txt); do
    for mode in ood iid; do
        for cell_type in $(cat datasets/sciplex3_partial/cell_types.txt); do
            echo \
                python ./scripts/train.py \
                --outdir ./results/sciplex3_partial/holdout-${drug}_${cell_type}/mode-${mode}/model-cellot \
                --config ./configs/tasks/sciplex3_partial.yaml \
                --config ./configs/models/cellot.yaml \
                --config.data.path ./datasets/sciplex3_partial/sciplex3_partial.h5ad \
                --config.data.target $drug \
                --config.datasplit.holdout ${drug}_${cell_type},control_${cell_type} \
                --config.datasplit.key drug_cell_type \
                --config.datasplit.mode $mode \
                --config.data.ae_emb.path ./results/sciplex3_partial/holdout-${drug}_${cell_type}/mode-${mode}/model-scgen \
                --restart=True;
        done;
    done;
done;