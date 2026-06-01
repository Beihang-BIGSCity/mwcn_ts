export CUDA_VISIBLE_DEVICES=0
python train2.py --gpu=0 --pred_len=96 --output_size=96
python train2.py --gpu=0 --pred_len=192 --output_size=192
python train2.py --gpu=0 --pred_len=336 --output_size=336
python train2.py --gpu=0 --pred_len=720 --output_size=720