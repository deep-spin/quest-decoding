

export STEPS=128
export TEMPERATURE=0.8


for BETA in 0.01 0.02 0.05 0.1 0.2 0.5 1
do
    for LLM in "alma" "tower" 
    do
        for LP in "en-de" "en-ru" "ru-en" "de-en"
        do
            python wmt_experiment.py --beta $BETA --temperature $TEMPERATURE --steps $STEPS --llm $LLM  --language_pair $LP --gpu_memory_utilization 0.6 --reward_batch_size 4 --save_path "mt-outputs/" --seed 0
        done
    done
done

