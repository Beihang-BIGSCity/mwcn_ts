#! /bin/bash
gpu=$1
exp_name=$2
wait_time=10
# dataset=('ETTh1' 'traffic')
# dataset=('exchange_rate')
dataset=('national_illness')

failed=()
for data in ${dataset[@]}
do
    for pred_len in 16 32 48 64
    do
        for seq_len in 32 48 64
        do
            if ! python train_tsm.py --lr 0.001 --dropout 0.0 --e_layers 2 --d_model 16 --seq_len $seq_len --pred_len $pred_len --output_size $pred_len --dataset $data --data_path ${data}.csv --gpu $gpu --log_dir ./log/tsmixer2_${data}_${exp_name}_${seq_len}_${pred_len}/ --cols OT --target OT; then
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