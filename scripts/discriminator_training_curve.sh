#!/bin/bash

echo export OMP_NUM_THREADS=4
echo export OPENBLAS_NUM_THREADS=4
echo export MKL_NUM_THREADS=4
echo export VECLIB_MAXIMUM_THREADS=4
echo export NUMEXPR_NUM_THREADS=4
echo export NUMBA_NUM_THREADS=4

for seed in 1327 1337 1347; do
    for use_discriminator in 1 0; do

        if [ "${use_discriminator}" -eq 1 ]; then
            model_name="scPILOT"
        else
            model_name="scPILOT_w_o_discriminator"
        fi

        patient_log_dir="../Logs/discriminator_training_curves/across_patients/${model_name}/seed-${seed}"
        mkdir -p "${patient_log_dir}"

        echo \
            CUDA_VISIBLE_DEVICES=1 \
            nohup python discriminator_training_curve.py \
            --benchmark across_patients \
            --query_key 101 \
            --seed "${seed}" \
            --split_seed 0 \
            --use_discriminator "${use_discriminator}" \
            \> "${patient_log_dir}/patient_101_${model_name}_seed${seed}.log" \
            2\>\&1 \
            \&;

        species_log_dir="../Logs/discriminator_training_curves/across_species/${model_name}/seed-${seed}"
        mkdir -p "${species_log_dir}"

        echo \
            CUDA_VISIBLE_DEVICES=1 \
            nohup python discriminator_training_curve.py \
            --benchmark across_species \
            --query_key mouse \
            --seed "${seed}" \
            --split_seed 0 \
            --use_discriminator "${use_discriminator}" \
            \> "${species_log_dir}/mouse_${model_name}_seed${seed}.log" \
            2\>\&1 \
            \&;
    done
done