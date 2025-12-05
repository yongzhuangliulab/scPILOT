for model_name in biolord identity scGen; do
    for query_key in $(cat ../Data/across_patients/sample_ids.txt); do
        echo \
            python across_patients_perturbation_prediction_other_models.py \
            --model_name ${model_name} \
            --query_key ${query_key};
    done;
done;