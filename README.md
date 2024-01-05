<div align="center">
<h1>
 DevOps-Model
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/codefuse-ai" target="_blank">Hugging Face</a> • 
🤖 <a href="https://modelscope.cn/organization/codefuse-ai" target="_blank">ModelScope</a> </p>

<div align="center">
<h4 align="center">
    <p>
        <b>中文</b> |
        <a href="https://github.com/codefuse-ai/CodeFuse-DevOps-Model/blob/main/README_EN.md">English</a>
    <p>
</h4>
</div>

DevOps-Model 是一系列业界首个开源的**中文开发运维大模型**，主要致力于在 DevOps 领域发挥实际价值。目前，DevOps-Model 能够帮助工程师回答在 DevOps 生命周期中遇到的问题。

我们基于 Qwen 系列模型，经过高质量中文 DevOps 语料加训后产出 **Base** 模型，然后经过 DevOps QA 数据对齐后产出 **Chat** 模型。我们的 Base 模型和 Chat 模型在开源和 DevOps 领域相关的评测数据上可以取得同规模模型中的**最佳效果**。欢迎来我们部署的在线试用地址试用模型效果：https://modelscope.cn/studios/codefuse-ai/DevOps-Model-Demo/summary
<br>
<br>
同时我们也在搭建 DevOps 领域专属的评测基准 [DevOpsEval](https://github.com/codefuse-ai/codefuse-devops-eval)，用来更好评测 DevOps 领域模型的效果。
<br>
<br>

# 最新消息
- [2023.12.22] 我们部署了 DevOps-Model 的在线模型问答地址，欢迎试用！！！ https://modelscope.cn/studios/codefuse-ai/DevOps-Model-Demo/summary
- [2023.12.06] 更新 Huggingface 下载地址
- [2023.10.31] 开源 DevOps-Model-14B Base 和 Chat 模型。
- [2023.10.30] 开源 DevOps-Model-7B Base 和 Chat 模型。


# 模型下载
开源模型和下载链接见下表：
🤗 Huggingface 地址

|         | 基座模型  | 对齐模型 | 对齐模型 Int4 量化 |
|:-------:|:-------:|:-------:|:-----------------:|
| 7B      |  [DevOps-Model-7B-Base](https://huggingface.co/codefuse-ai/CodeFuse-DevOps-Model-7B-Base)| [DevOps-Model-7B-Chat](https://huggingface.co/codefuse-ai/CodeFuse-DevOps-Model-7B-Chat) | Coming Soon|
| 14B     | [DevOps-Model-14B-Base](https://huggingface.co/codefuse-ai/CodeFuse-DevOps-Model-14B-Base) | [DevOps-Model-14B-Chat](https://huggingface.co/codefuse-ai/CodeFuse-DevOps-Model-14B-Chat) | Coming Soon |


🤖 ModelScope 地址

|         | 基座模型  | 对齐模型 | 对齐模型 Int4 量化 |
|:-------:|:-------:|:-------:|:-----------------:|
| 7B      |  [DevOps-Model-7B-Base](https://modelscope.cn/models/codefuse-ai/CodeFuse-DevOps-Model-7B-Chat/summary) | [DevOps-Model-7B-Chat](https://modelscope.cn/models/codefuse-ai/CodeFuse-DevOps-Model-7B-Chat/summary) | Coming Soon|
| 14B     | [DevOps-Model-14B-Base](https://modelscope.cn/models/codefuse-ai/CodeFuse-DevOps-Model-14B-Base/summary) | [DevOps-Model-14B-Chat](https://modelscope.cn/models/codefuse-ai/CodeFuse-DevOps-Model-14B-Chat/summary) | Coming Soon |


# 模型评测
我们先选取了 CMMLU 和 CEval 两个评测数据集中和 DevOps 相关的一共六项考试。总计一共 574 道选择题，具体信息如下：

|  评测数据集 | 考试科目  | 题数  | 
|:-------:|:-------:|:-------:|
|   CMMLU  | Computer science | 204 |
|   CMMLU  | Computer security | 171 |
|   CMMLU  | Machine learning | 122 |
| CEval   | College programming | 37 |
| CEval   | Computer architecture | 21 |
| CEval   | Computernetwork | 19 |


我们分别测试了 Zero-shot 和 Five-shot 的结果，我们的 7B 和 14B 系列模型可以取得在测试的模型中最好的成绩，更多的测试后续也会放出。

|Base 模型|Zero-shot 得分|Five-shot 得分|
|:-------:|:-------:|:-------:|
|**DevOps-Model-14B-Base**| **70.73** | **73.00** |
|Qwen-14B-Base| 69.16 | 71.25  |
|Baichuan2-13B-Base| 55.75 | 61.15 |
|**DevOps-Model-7B-Base**| **62.72** | **62.02** |
|Qwen-7B-Base| 55.75 | 56.00 | 
|Baichuan2-7B-Base| 49.30 | 55.4 |
|Internlm-7B-Base| 47.56 | 52.6 |
<br>

|Chat 模型|Zero-shot 得分|Five-shot 得分|
|:-------:|:-------:|:-------:|
|**DevOps-Model-14B-Chat**| **74.04** | **75.96** |
|Qwen-14B-Chat| 69.16 | 70.03 |
|Baichuan2-13B-Chat| 52.79 | 55.23 |
|**DevOps-Model-7B-Chat**| **62.20** | **64.11** |
|Qwen-7B-Chat| 46.00 | 52.44 |
|Baichuan2-7B-Chat| 52.56 | 55.75 |
|Internlm-7B-Chat| 52.61 | 55.75 |

<br>
 <br>

# 快速使用
我们提供简单的示例来说明如何利用 🤗 Transformers 快速使用 Devops-Model-Chat 模型。

## 安装依赖

```bash
pip install -r requirements.txt
```

## Chat 模型推理示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("path_to_DevOps-Model-Chat", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("path_to_DevOps-Model-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()

# 指定 generation_config
model.generation_config = GenerationConfig.from_pretrained("path_to_DevOps-Model-Chat", trust_remote_code=True)

# 第一轮对话
resp, hist = model.chat(query='你是谁', tokenizer=tokenizer, history=None)
print(resp)
# 我是 DevOps-Model，一个由蚂蚁集团平台技术事业群风险智能团队和北京大学联合研发的人工智能机器人，可以与用户进行自然语言交互，并协助解答 DevOps 全生命周期中的各种问题。如果您有任何需要协助的问题或者想要进行闲聊，都可以和我交流哦。

# 第二轮对话
resp2, hist2 = model.chat(query='Java 中 HashMap 和 Hashtable 有什么区别', tokenizer=tokenizer, history=hist)
print(resp2)
# HashMap 和 Hashtable 都是 Java 中常用的哈希表实现，它们的主要区别在于：
# 1. Hashtable 是线程安全的，而 HashMap 不是线程安全的，因此在多线程环境下，Hashtable 的性能更稳定。
# 2. Hashtable 中的方法都是同步的，而 HashMap 的方法不是同步的，因此在多线程环境下，Hashtable 的性能更好。
# 3. Hashtable 中的 key 和 value 都必须实现 Serializable 接口，而 HashMap 中的 key 和 value 可以是任何对象，包括基本数据类型。
# 4. Hashtable 的初始容量是 11，而 HashMap 的初始容量是 16。
# 总之，如果需要在多线程环境下使用哈希表，并且需要保证线程安全，那么应该使用 Hashtable；如果不需要考虑线程安全，或者需要快速地进行哈希表操作，那么应该使用 HashMap。

# 第三轮对话
resp3, hist3 = model.chat(query='线程安全代表什么', tokenizer=tokenizer, history=hist2)
print(resp3)
# 线程安全是指在多线程环境下，程序能够正确地处理并发访问，并且不会出现数据竞争、死锁、饥饿等异常情况。线程安全的程序可以保证在不同的线程之间共享同一个数据结构时，数据的正确性和一致性。线程安全的实现通常需要使用同步机制，如锁、原子操作等，来保证对共享数据的访问是线程安全的。在 Java 中，可以通过 synchronized 关键字、Lock 接口等机制来实现线程安全。
```


# 模型训练

## 数据准备
代码内部通过调用 datasets.load_dataset 读取数据，支持 load_dataset 所支持的数据读取方式，比如 json，csv，自定义读取脚本等方式（但推荐数据准备为 jsonl 格式的文件）。然后还需要更新 `data/dataset_info.json` 文件，具体可以参考 `data/README.md`。

## 预训练
如果收集了一批文档之类的语料（比如公司内部产品的文档）想要在 devopspal 模型上加训，可以执行 `scripts/devops-model-pt.sh` 来发起一次加训来让模型学习到这批文档的知识，具体代码如下:

```bash
set -v 

torchrun --nproc_per_node=8 --nnodes=$WORLD_SIZE --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR --node_rank=$RANK src/train_bash.py \
    --deepspeed conf/deepspeed_config.json \    # deepspeed 配置地址
	--stage pt \    # 代表执行 pretrain
    --model_name_or_path path_to_model \    # huggingface下载的 devopspal 模型地址
    --do_train \
    --report_to 'tensorboard' \
    --dataset your_corpus \    # 数据集名字，要和在 dataset_info.json 中定义的一致
    --template default \    # template，pretrain 就是 default
    --finetuning_type full \  # 全量或者 lora
    --output_dir path_to_output_checkpoint_path \    # 模型 checkpoint 保存的路径
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --evaluation_strategy steps \
    --logging_steps 10 \
    --max_steps 1000 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --learning_rate 5e-6 \
    --plot_loss \
    --max_source_length=2048 \
    --dataloader_num_workers 8 \
    --val_size 0.01 \
    --bf16 \
    --overwrite_output_dir
```

使用者可以在这个基础上调整来发起自己的训练，更加详细的可配置项建议通过 `python src/train_bash.py -h` 来获取完整的参数列表。

## 指令微调
如果收集了一批 QA 数据想要针对 devopspal 再进行对齐的话，可以执行 `scripts/devops-model-sft.sh` 来发起一次加训来让模型在收集到的模型上进行对齐，具体代码如下:
```bash
set -v 

torchrun --nproc_per_node=8 --nnodes=$WORLD_SIZE --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR --node_rank=$RANK src/train_bash.py \
    --deepspeed conf/deepspeed_config.json \    # deepspeed 配置地址
    --stage sft \    # 代表执行 pretrain
    --model_name_or_path path_to_model \    # huggingface下载的模型地址
    --do_train \
    --report_to 'tensorboard' \
    --dataset your_corpus \    # 数据集名字，要和在 dataset_info.json 中定义的一致
    --template chatml \    # template qwen 模型固定写 chatml
    --finetuning_type full \    # 全量或者 lora
    --output_dir /mnt/llm/devopspal/model/trained \     # 模型 checkpoint 保存的路径
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --evaluation_strategy steps \
    --logging_steps 10 \
    --max_steps 1000 \
    --save_steps 100 \
    --eval_steps 100 \
    --learning_rate 5e-5 \
    --plot_loss \
    --max_source_length=2048 \
    --dataloader_num_workers 8 \
    --val_size 0.01 \
    --bf16 \
    --overwrite_output_dir
```

使用者可以在这个基础上调整来发起自己的 SFT 训练，更加详细的可配置项建议通过 `python src/train_bash.py -h` 来获取完整的参数列表。

## 量化
我们将会提供了 DevOps-Model-Chat 系列的量化模型，当然也可以通过以下代码来量化自己加训过的模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model
import torch

# 加载模型
model_name = "path_of_your_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# 加载数据
# todo

# 开始量化
quantizer = GPTQQuantizer(bits=4, dataset="c4", block_name_to_quantize = "model.decoder.layers", model_seqlen = 2048)
quantized_model = quantizer.quantize_model(model, tokenizer)

# 保存量化后的模型
out_dir = 'save_path_of_your_quantized_model'
quantized_model.save_quantized(out_dir)
```


# 联系我们
![](https://github.com/codefuse-ai/CodeFuse-DevOps-Model/blob/main/imgs/wechat2.png)


# 免责声明
由于语言模型的特性，模型生成的内容可能包含幻觉或者歧视性言论。请谨慎使用 DevOps-Model 系列模型生成的内容。
如果要公开使用或商用该模型服务，请注意服务方需承担由此产生的不良影响或有害言论的责任，本项目开发者不承担任何由使用本项目（包括但不限于数据、模型、代码等）导致的危害或损失。

# 引用
如果使用本项目的代码或模型，请引用本项目论文：

链接：[DevOps-Model](https://arxiv.org)

```
@article{devopspal2023,
  title={},
  author={},
  journal={arXiv preprint arXiv},
  year={2023}
}
```

# 致谢
本项目参考了以下开源项目，在此对相关项目和研究开发人员表示感谢。
- [LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning)
- [QwenLM](https://github.com/QwenLM)
