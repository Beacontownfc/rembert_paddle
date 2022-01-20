# RemBERT paddle实现

## 1 简介 
本项目基于paddlepaddle框架复现了RemBERT预训练模型，主要复现XNLI和PAWS-X数据集的结果。

**项目说明：**

- 项目参考
[https://github.com/huggingface/transformers/tree/master/src/transformers/models/rembert](https://github.com/huggingface/transformers/tree/master/src/transformers/models/rembert)
- 本项目RemBERT的tokenizer基于paddlenlp复现
- 本项目的模型基于paddlepaddle框架复现

## 2 复现精度
>#### 在PAWS-X数据集的测试结果如下表。

|网络 |opt|batch_size|数据集|ACC|
| :---: | :---: | :---: | :---: | :---: |
|RemBERT|AdamW|64|PAWS-X|94.95|

在PAWS-X数据集上的训练日志:
[PAWS-X_train.log](train.log)

>#### 在XNLI数据集的测试结果如下表。

|网络 |opt|batch_size|数据集|ACC|
| :---: | :---: | :---: | :---: | :---: |
|RemBERT|AdamW|16|XNLI||

在XNLI数据集上的训练日志:
[XNLI_train.log](train.log)

## 3 数据集
XNLI和PAWS-X是XTREME数据集上的sequence pair classification任务

XNLI数据集下载地址：[XNLI下载](https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip) （若您的网络不佳，可使用在此处
[下载地址](https://aistudio.baidu.com/aistudio/datasetdetail/126002)
下载，multinli.train.en.tsv、 xnli.dev.en.tsv和xnli.test.en.tsv 分别为XNLI的训练集、验证集和测试集）

PAWS-X数据集下载地址: [PAWS-X下载](https://storage.googleapis.com/paws/pawsx/x-final.tar.gz) （若您的网络不佳，可使用在此处
[下载地址](https://aistudio.baidu.com/aistudio/datasetdetail/126002)
下载，train.tsv、 dev_2k.tsv和test_2k.tsv 分别为PAWS-X的训练集、验证集和测试集）

## 4环境依赖
运行以下命令即可配置环境
```bash
pip install -r requirements.txt
```

## 5快速开始
#### 数据集下载好后，解压数据集, 同时下载预训练权重: [下载地址](https://aistudio.baidu.com/aistudio/datasetdetail/125938)

若希望训练并评估模型在XNLI测试集的精度，运行如下命令:
```bash
python main.py --task=xnli --data_dir=<DATA_DIR> --output_dir=<OUTPUT_DIR> --pretrain_model=<MODEL_DIR>
```
<DATA_DIR>、<OUTPUT_DIR>和<MODEL_DIR>是你的存放数据集的文件夹、输出文件夹和预训练模型文件夹，推荐使用绝对路径

运行结束后你将看到如下结果:
```bash
ACC: 94.95
```

若希望训练并评估模型在PAWS-X测试集的精度，运行如下命令:
```bash
python main.py --task=paws --data_dir=<DATA_DIR> --output_dir=<OUTPUT_DIR> --pretrain_model=<MODEL_DIR>
```

运行结束后你将看到如下结果:
```bash
ACC: 94.95
```
## 6 代码结构与详细说明
```
├─data
| ├─sentencepiece.model      #vocab文件
├─rembert
| ├─model_utils.py           #rembert模型文件
| ├─rembert_model.py         #rembert模型文件
| ├─rembert_tokenizer        #tokenizer  
| ├─rembertConfig.py         #rembert配置文件                     
├─datagenerator.py           #数据生成器
├─dataProcessor.py           #数据处理
├─main.py                    #训练并测试模型
├─rembertForSeqPairPred.py   #rembert下游任务
├─trainer.py                 #训练文件                           
```