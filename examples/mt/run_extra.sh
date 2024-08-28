


export STEPS=128
export TEMPERATURE=0.8


for BETA in 0.02 0.05 0.2 0.5 # 0.01 0.1 1 
do
    for LLM in "alma" 
    do
        for LP in "en-zh" "en-cs"
        do
            python wmt_experiment.py --beta $BETA --temperature $TEMPERATURE --steps $STEPS --llm $LLM  --language_pair $LP --gpu_memory_utilization 0.95 --reward_device 1 --reward_batch_size 64 --save_path "mt-outputs/" --seed 0
        done
    done
done

