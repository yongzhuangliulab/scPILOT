for query_key in $(cat ../Data/across_cell_lines/cell_types.txt); do
    echo \
        python across_cell_lines_perturbation_prediction.py \
        --query_key ${query_key};
done;