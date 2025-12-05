#! /bin/bash

for mode in ood; do
    for target_gene_cell_type in $(cat ../Data/across_cell_lines/target_gene_cell_types.txt); do
        for model in scgen; do
            echo \
                python cellot_train.py \
                --outdir ../cellot_results/across_cell_lines/holdout-${target_gene_cell_type}/mode-${mode}/model-${model} \
                --config ../cellot_configs/tasks/across_cell_lines.yaml \
                --config ../cellot_configs/models/${model}.yaml \
                --config.data.path ../Data/across_cell_lines/IFNGR2.h5ad \
                --config.datasplit.holdout $target_gene_cell_type \
                --config.datasplit.key target_gene_cell_type \
                --config.datasplit.mode $mode \
                --restart=True;
        done;
    done;
done;


# This is to use the same encoder as scgen
for mode in ood; do
    for cell_type in $(cat ../Data/across_cell_lines/cell_types.txt); do
        echo \
            python cellot_train.py \
            --outdir ../cellot_results/across_cell_lines/holdout-${cell_type}/mode-${mode}/model-cellot \
            --config ../cellot_configs/tasks/across_cell_lines.yaml \
            --config ../cellot_configs/models/cellot.yaml \
            --config.data.path ../Data/across_cell_lines/IFNGR2.h5ad \
            --config.datasplit.holdout $cell_type \
            --config.datasplit.key cell_type \
            --config.datasplit.mode $mode \
            --config.data.ae_emb.path ../cellot_results/across_cell_lines/holdout-IFNGR2_${cell_type}/mode-${mode}/model-scgen \
            --restart=True;
    done;
done;