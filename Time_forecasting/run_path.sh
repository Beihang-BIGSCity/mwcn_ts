#! /bin/bash
gpu=$1
exp_name=$2
wait_time=10
# dataset=('ETTh1' 'traffic')
dataset=('traffic')
# dataset=('ETTh1')
seq_len=()

failed=()
for data in ${dataset[@]}
do
    if ! python train_pa.py --seq_len 96 --pred_len 96 --output_size 96 --dataset $data --data_path ${data}.csv --gpu $gpu --log_dir ./log/path_${data}_${exp_name}_96/ --cols OT --target OT --patch_size_list 24 12 8 4 12 8 6 4 8 6 4 2 ; then
        failed+=("$data") 
    fi

    if ! python train_pa.py --seq_len 192 --pred_len 192 --output_size 192 --dataset $data --data_path ${data}.csv --gpu $gpu --log_dir ./log/path_${data}_${exp_name}_192/ --cols OT --target OT --patch_size_list 24 12 8 16 12 8 6 4 8 6 4 2 ; then
        failed+=("$data") 
    fi

    if ! python train_pa.py --seq_len 336 --pred_len 336 --output_size 336 --dataset $data --data_path ${data}.csv --gpu $gpu --log_dir ./log/path_${data}_${exp_name}_336/ --cols OT --target OT --patch_size_list 12 7 4 21 7 12 4 21 12 7 12 4  ; then
        failed+=("$data") 
    fi

    if ! python train_pa.py --seq_len 720 --pred_len 720 --output_size 720 --dataset $data --data_path ${data}.csv --gpu $gpu --log_dir ./log/path_${data}_${exp_name}_720/ --cols OT --target OT --patch_size_list 18 20 36 9 4 30 4 18 20 9 18 4  ; then
        failed+=("$data") 
    fi

    if [ $wait_time -gt 0 ]; then  
        sleep $wait_time  
    fi  
done

if [ ${#failed[@]} -gt 0 ]; then
    echo "以下数据集训练失败："  
    for f in "${failed[@]}"; do  
        echo "$f"  
    done  
fi