for model_name in biolord identity scGen; do
    for query_key in $(cat ../Data/across_species/species.txt); do
        echo \
            python across_species_perturbation_prediction_other_models.py \
            --model_name ${model_name} \
            --query_key ${query_key};
    done;
done;