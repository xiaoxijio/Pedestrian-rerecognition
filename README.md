原代码链接：**[Relation-Aware-Global-Attention-Networks](https://github.com/microsoft/Relation-Aware-Global-Attention-Networks)**  
因为使用源代码在windows上运行大概率会报错，所以我修改了一些地方，添加了一些注释，方便理解  
数据链接：晚上回宿舍上传，着急的话可以自己去上面链接找，需要data数据和weights数据  
训练参数：  
```
-a resnet50_rga
-b 16
-d cuhk03labeled
-j 0
--opt adam
--dropout 0
--combine-trainval
--seed 16
--num_gpu 1
--epochs 600
--features 2048
--start_save 300
--branch_name rgasc
--data-dir ./data
--logs-dir ./logs/RGA-SC/cuhk03labeled_b64f2048
```
测试参数：  
```
-a resnet50_rga
-b 16
-d cuhk03labeled
-j 0
--opt adam
--dropout 0
--combine-trainval
--seed 16
--num_gpu 1
--epochs 600
--features 2048
--start_save 300
--branch_name rgasc
--data-dir ./data
--logs-dir ./logs/RGA-SC/cuhk03labeled_b64f2048
--evaluate
--resume ./logs/RGA-SC/cuhk03labeled_b64f2048/checkpoint_600.pth.tar
```
