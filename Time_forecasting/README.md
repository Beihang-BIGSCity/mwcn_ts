README in Time_forecasting:<br>
parameter adjust<br>
lr:learning rate,default:0.001<br>
log_dir: the folder save the results<br>
dataset:dataset #ECL ETT weather...<br>
root_path: save all the datasets<br>
data_path:数据集文件名,default='ECL.csv'.<br>
(root_path+data_path is the total data path)<br>
seq_len: input length, default:96<br>
label_len: length of label, default:0<br>
pred_len:predict length, default:96<br>
output_size:output length, default:96<br>
features: forecasting task, default='S', options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')<br>
target:'target feature in S or MS task'

