for query_key in $(cat ../Data/across_cell_types/cell_labels.txt); do
    echo \
        python across_cell_types_perturbation_prediction.py \
        --query_key ${query_key};
done;