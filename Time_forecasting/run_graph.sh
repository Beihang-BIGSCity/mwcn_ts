#! /bin/bash
gpu=$1
exp_name=$2
wait_time=10
# dataset=('ETTh1' 'traffic')
dataset=('weather' 'exchange_rate')
seq_len=()

failed=()
for data in ${dataset[@]}
do
    for pred_len in 192 336 96 720 
    do
        for seq_len in 96 192 336 720
        do
            if ! python train2.py --seq_len $seq_len --pred_len $pred_len --output_size $pred_len --dataset $data --data_path ${data}.csv --gpu $gpu --log_dir ./log/VGG_${data}_${exp_name}_${seq_len}_${pred_len}/ --cols OT --target OT; then
                failed+=("$data") 
            fi
            if [ $wait_time -gt 0 ]; then  
                sleep $wait_time  
            fi  
        done
    done
done

if [ ${#failed[@]} -gt 0 ]; then
    echo "以下数据集训练失败："  
    for f in "${failed[@]}"; do  
        echo "$f"  
    done  
fi