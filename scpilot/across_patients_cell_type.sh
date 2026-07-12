#!/bin/bash

echo export OMP_NUM_THREADS=4
echo export OPENBLAS_NUM_THREADS=4
echo export MKL_NUM_THREADS=4
echo export VECLIB_MAXIMUM_THREADS=4
echo export NUMEXPR_NUM_THREADS=4
echo export NUMBA_NUM_THREADS=4

for query_key in $(cat ../Data/across_patients/sample_ids.txt); do
    log_dir="../Logs/across_patients/cell_type/query-${query_key}"
    mkdir -p "${log_dir}"

    echo \
        nohup python across_patients_cell_type.py \
        --query_key ${query_key} \
        --seed all \
        \> "${log_dir}/across_patients_cell_type_${query_key}.log" \
        2\>\&1 \
        \&;
done