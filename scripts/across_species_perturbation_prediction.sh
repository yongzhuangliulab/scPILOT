for query_key in $(cat ../Data/across_species/species.txt); do
    echo \
        python across_species_perturbation_prediction.py \
        --query_key ${query_key};
done;