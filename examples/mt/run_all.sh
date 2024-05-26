


export STEPS=128
export TEMPERATURE=0.8

for BETA in 0.05 0.1 0.5 1
do
    for LLM in "alma" "tower"
    do
        for LP in "en-de" "en-ru" "ru-en" "de-en"
        do
        
            export FILE_NAME="wmt23_"$LLM"_"$LP"_beta_"$BETA"_temperature_"$TEMPERATURE"_steps_"$STEPS".json"

            python wmt23_experiment.py --beta $BETA --temperature $TEMPERATURE --steps $STEPS --llm $LLM  --language_pair $LP --gpu_memory_utilization 0.6 --reward_batch_size 8 --output_file_name $FILE_NAME
        
        done
    done
done
