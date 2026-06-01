Readme
1.运行python train2.py,运行结果和运行的代码可以在log_dir里面查看。
2.可设置参数介绍
lr:learning rate,default:0.001
log_dir:存放实验结果
dataset:dataset #ECL ETT weather...
root_path:数据集目录路径,可选择路径：./Datasets_informer/和./Datasets_autoformer。其中Datasets_informer包含ECL, ETT, weather数据集, Datasets_autoformer包含exchange_rate, illness, traffic数据集.
data_path:数据集文件名,default='ECL.csv'.
(root_path+data_path为最终的完整数据集路径.)
seq_len: input length, default:96
label_len: length of label, default:0
pred_len:predict length, default:96
output_size:output length, default:96
features: forecasting task, default='S', options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
target:'target feature in S or MS task'
3.scripts的文件可以运行对应的数据集
