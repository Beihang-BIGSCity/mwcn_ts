#README in Time_classification..
run bash run_layers.sh..
main parameters:..
lr:learning rate..
log_dir: folder to save results..
dataset: data20240812/UCRArchieve_2018..
batch_size: ..
optimizer: adam or momentum..

可以在命令行更改参数:python train1.py --lr=2e-4..
模型选择:net=conv2d_multiwave,conv2d_multiwave_v2_att22等，位于train1.py的123行。可根据注释选择。
