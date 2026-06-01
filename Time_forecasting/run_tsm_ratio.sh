#! /bin/bash
gpu=$1
wait_time=10
dataset=('ETTh1' 'traffic' 'electricity')
ratios=(1 2 3 4 5)
# dataset=('exchange_rate')
# dataset=('electricity')
seq_len=()

failed=()
for data in ${dataset[@]}
do
    for pred_len in 96 192 336 720
    do
        for seq_len in 336
        do
            for ratio in ${ratios[@]}
            do
                if ! python train_tsm.py --ratio ${ratio} --num_layers 3 --dropout 0 --e_layers 2 --d_model 16 --seq_len $seq_len --pred_len $pred_len --output_size $pred_len --dataset $data --data_path ${data}.csv --gpu $gpu --log_dir ./log/tsmixer_${data}_ratio${ratio}_${seq_len}_${pred_len}/ --cols OT --target OT; then
                    failed+=("$data") 
                fi
                if [ $wait_time -gt 0 ]; then  
                    sleep $wait_time  
                fi
            done
        done
    done
done

if [ ${#failed[@]} -gt 0 ]; then
    echo "以下数据集训练失败："  
    for f in "${failed[@]}"; do  
        echo "$f"  
    done
fi