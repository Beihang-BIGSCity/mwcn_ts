#readme
使用方式 python train1.py
可使用的主要参数：
lr:learning rate
log_dir:保存代码文件夹
dataset:数据集,数据集具体名称见data20240812/UCRArchieve_2018下面
batch_size: 批尺寸
optimizer:学习策略,adam or momentum

可以在命令行更改参数:python train1.py --lr=2e-4
模型选择:net=conv2d_multiwave,conv2d_multiwave_v2_att22等，位于train1.py的123行。可根据注释选择。