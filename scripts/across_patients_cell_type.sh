for query_key in $(cat ../Data/across_patients/sample_ids.txt); do
    echo \
        python across_patients_cell_type.py \
        --query_key ${query_key};
done;