#README in Time_classification<br>
run bash run_layers.sh<br>
main parameters:<br>
lr:learning rate<br>
log_dir: folder to save results<br>
dataset: data20240812/UCRArchieve_2018<br>
batch_size: <br>
optimizer: adam or momentum<br>

可以在命令行更改参数:python train1.py --lr=2e-4<br>
模型选择:net=conv2d_multiwave,conv2d_multiwave_v2_att22等，位于train1.py的123行。可根据注释选择。
