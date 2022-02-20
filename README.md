# RemBert Paddle

## 1 简介 

**本项目基于PaddlePaddle复现的RemBert，完成情况如下:**

- 在XTREME数据集上的xnli和paws-x任务均达到论文精度
- RemBertTokenizer已基于paddlenlp进行复现
- 我们复现的RemBert也是基于paddlenlp
- 我们提供aistudio notebook, 帮助您快速验证模型

**项目参考：**
- [https://github.com/huggingface/transformers/tree/master/src/transformers/models/rembert](https://github.com/huggingface/transformers/tree/master/src/transformers/models/rembert)


## 2 复现精度
>#### 在XTREME的PAWS-X数据集的测试效果如下表。
>ACC是准确率的简写

|网络 |opt|batch_size|数据集|ACC|ACC(原论文)|
| :---: | :---: | :---: | :---: | :---: | :---: |
|RemBert|AdamW|16|XTREME-PAWS-X|87.78|87.5|

>复现代码训练日志：
[复现代码训练日志](paws.log)
>
>#### 在XTREME-XNLI数据集的测试效果如下表。

|网络 |opt|batch_size|数据集|ACC|ACC(原论文)|
| :---: | :---: | :---: | :---: | :---: | :---: |
|RemBert|AdamW|16|XTREME-XNLI|80.89|80.8|

>复现代码及训练日志：
[复现代码训练日志](xnli.log)

>验收标准为在XTREME的PAWS-X数据集和XTREME-XNLI数据集的测试精度的平均值为84.2
>
>`原论文: (87.5 + 80.8) / 2 = 84.15`
>
>`复现精度：(87.78 + 80.89) / 2 = 84.33`

## 3 数据集
XTREME的XNLI和PAWS-X数据集是XTREME上的sequence-pair classification任务

下载XTREME-XNLI数据集:
训练集:[下载地址](https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip)
测试集:[下载地址](https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip)
其中训练集为位于`XNLI-MT-1.0/multinli/multinli.train.en.tsv`, 测试集位于`XNLI-1.0/xnli.test.tsv`

下载XTREME-PAWS-X数据集：
[下载地址](https://storage.googleapis.com/paws/pawsx/x-final.tar.gz)
训练集、验证集和测试集分别为`train`、`dev`和`test`开头的`tsv`文件,

`我们已经把所有的语言合进测试集, 此处可下载`[下载地址](https://aistudio.baidu.com/aistudio/datasetdetail/126002)，`test_2k.tsv是我们合并了多语言的测试集`


## 4环境依赖
运行以下命令即可配置环境
```bash
pip install paddlenlp==2.2.4
```

## 5 快速开始
如果你觉得以下步骤过于繁琐，您可以直接到此处
[链接](https://aistudio.baidu.com/aistudio/projectdetail/3426124)
利用我们提供的AISTUDIO NOTEBOOK快速验证和训练评估模型。

首先，您需要下载预训练权重:
[下载地址](https://aistudio.baidu.com/aistudio/datasetdetail/125938)

checkpoint下载地址:
paws-x数据集checkpoint:
[下载地址](https://aistudio.baidu.com/aistudio/datasetdetail/126525)

xnli数据集checkpoint:
[下载地址](https://aistudio.baidu.com/aistudio/datasetdetail/127104)

###### 训练并测试在XTREME-XNLI数据集上的ACC：


```bash
python main.py --task=xnli --do_train=1 --do_eval=1 --data_dir=<DATA_DIR> --output_dir=<OUTPUT_DIR> --pretrain_model=<MODEL_DIR> --learning_rate=1e-5
```

从checkpoint中快速评估模型:
```bash
python main.py --task=xnli --do_eval=1 --data_dir=<DATA_DIR> --output_dir=<OUTPUT_DIR>
```

说明：

- `<DATA_DIR>`、`<OUTPUT_DIR>`和`<MODEL_DIR>`分别为数据集文件夹路径、输出文件夹路径和预训练权重文件夹路径

运行结束后你将看到如下结果:
```bash
Acc 80.60
```

###### 训练并测试在XTREME-PAWS-X数据集上的ACC：

```bash
python main.py main.py --task=paws --do_train=1 --do_eval=1 --data_dir=<DATA_DIR> --output_dir=<OUTPUT_DIR> --eval_step=500 --pretrain_model=<MODEL_DIR>
```

从checkpoint中快速评估模型:
```bash
python main.py --task=paws --do_eval=1 --data_dir=<DATA_DIR> --output_dir=<OUTPUT_DIR>
```

运行结束后你将看到如下结果:
```bash
ACC 87.78
```

## 6 代码结构与详细说明
```

├─data                     # 词库文件夹
| ├─sentencepiece.model    # tokenizer 词库文件
├─rembert                  # RemBert模型文件夹
| ├─rembert_model.py       # RemBert模型
| ├─rembert_tokenizer.py   # tokenizer文件
├─datagenerator.py         # data生成器
├─dataProcessor.py         # 数据生成器 
├─main.py                  # 主文件
├─trainer.py               # 训练文件                                    
```
