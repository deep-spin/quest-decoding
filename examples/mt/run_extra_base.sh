


STEPS=128
#TEMPERATURE=0.8
YEAR=23

for TEMP in 0.2 0.3 0.4 0.6 0.7 0.9 1.0 # 0.1, 0.5, 0.8
do
    for LLM in "alma" 
    do
        for LP in "en-zh" "en-cs"
        do
            python wmt_baseline.py --temperature $TEMP --steps $STEPS --llm $LLM  --language_pair $LP --gpu_memory_utilization 0.95 --save_path "mt-outputs/" --seed 0 --year $YEAR --device_count 4
        done
    done
done


