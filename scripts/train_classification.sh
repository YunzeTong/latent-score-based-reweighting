            
datasets=("adult") 

timeweights=("EDM") 

temperatures=(3) 

for ((j=0; j<${#timeweights[@]}; j++))
do
        dataset=${datasets[j]}
        timeweight=${timeweights[j]}
        T=${temperatures[j]}
        CUDA_VISIBLE_DEVICES=3 python ./classification_process/main.py --dataname $dataset \
            --log_path ./classification_process/discrete_log \
            --method several_sigma_error_diff --mode train --comment several_sigma_error_diff \
            --use_weight --weight_criterion several_timestep_error_diff \
            --timestep_weight_criterion $timeweight --temperature $T --error_reflection softmax \
            --seed 42
done