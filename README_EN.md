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
        <a href="https://github.com/codefuse-ai/CodeFuse-DevOps-Model/blob/main/README.md">中文</a> |
	<b>English</b> 
    <p>
</h4>
</div>

DevOps-Model is a Chinese **DevOps large model**, mainly dedicated to exerting practical value in the field of DevOps. Currently, DevOps-Model can help engineers answer questions encountered in the all DevOps life cycle.

Based on the Qwen series of models, we output the **Base** model after additional training with high-quality Chinese DevOps corpus, and then output the **Chat** model after alignment with DevOps QA data. Our Base model and Chat model can achieve the best results among models of the same scale based on evaluation data related to the DevOps fields.

<br>
At the same time, we are also building an evaluation benchmark [DevOpsEval](https://github.com/codefuse-ai/codefuse-devops-eval) exclusive to the DevOps field to better evaluate the effect of the DevOps field model.
<br>
<br>

# Update
- [2023.10.30] Open source DevOps-Model-7B Base and Chat models.


# Download
Open source models and download links are shown in the table below:
🤗 Huggingface 

|         | Base Model  | Chat Model | Chat Model(Int4) |
|:-------:|:-------:|:-------:|:-----------------:|
| 7B      | Coming Soon | Coming Soon| Coming Soon|
| 14B     | Coming Soon | Coming Soon| Coming Soon |

🤖 ModelScope 

|         | Base Model  | Chat Model | Chat Model(Int4) |
|:-------:|:-------:|:-------:|:-----------------:|
| 7B      |  [DevOps-Model-7B-Base](https://modelscope.cn/models/codefuse-ai/CodeFuse-DevOps-Model-7B-Chat/summary) | [DevOps-Model-7B-Chat](https://modelscope.cn/models/codefuse-ai/CodeFuse-DevOps-Model-7B-Chat/summary) | Coming Soon|
| 14B     | Coming Soon | Coming Soon | Coming Soon |


# Evaluation
We first selected a total of six exams related to DevOps in the two evaluation data sets of CMMLU and CEval. There are a total of 574 multiple-choice questions. The specific information is as follows:

| Evaluation dataset | Exam subjects | Number of questions |
|:-------:|:-------:|:-------:|
|   CMMLU  | Computer science | 204 |
|   CMMLU  | Computer security | 171 |
|   CMMLU  | Machine learning | 122 |
| CEval   | College programming | 37 |
| CEval   | Computer architecture | 21 |
| CEval   | Computernetwork | 19 |


We tested the results of Zero-shot and Five-shot respectively. Our 7B and 14B series models can achieve the best results among the tested models. More tests will be released later.

|Base Model|Zero-shot Score|Five-shot Score|
|:-------:|:-------:|:-------:|
|**DevOps-Model-14B-Base**| **70.73** | **73.00** |
|Qwen-14B-Base| 69.16 | 71.25  |
|Baichuan2-13B-Base| 55.75 | 61.15 |
|**DevOps-Model-7B-Base**| **62.72** | **62.02** |
|Qwen-7B-Base| 55.75 | 56.00 | 
|Baichuan2-7B-Base| 49.30 | 55.4 |
|Internlm-7B-Base| 47.56 | 52.6 |
<br>

|Chat Model|Zero-shot Score|Five-shot Score|
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

# Quickstart
We provide simple examples to illustrate how to quickly use Devops-Model-Chat models with 🤗 Transformers.

## Requirement

```bash
pip install -r requirements.txt
```

## Chat Model Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("path_to_DevOps-Model-Chat", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("path_to_DevOps-Model-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()

model.generation_config = GenerationConfig.from_pretrained("path_to_DevOps-Model-Chat", trust_remote_code=True)

resp2, hist2 = model.chat(query='What is the difference between HashMap and Hashtable in Java', tokenizer=tokenizer, history=hist)
```


# Model Finetune

## Data
The code internally reads data by calling `datasets.load_dataset`, and supports the data reading methods supported by `load_dataset`, such as json, csv, custom reading scripts, etc. (but it is recommended that the data be prepared in jsonl format files). Then you also need to update the `data/dataset_info.json` file. For details, please refer to `data/README.md`.


## Pretrain
If you have collected a batch of documents and other corpus (such as company internal product documents) and want to train based on our model, you can execute `scripts/devops-model-pt.sh` to initiate an additional training to let the model learn The specific codes for the knowledge of this batch of documents are as follows:

```bash
set -v 

torchrun --nproc_per_node=8 --nnodes=$WORLD_SIZE --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR --node_rank=$RANK src/train_bash.py \
    --deepspeed conf/deepspeed_config.json 
	--stage pt \
    --model_name_or_path path_to_model \
    --do_train \
    --report_to 'tensorboard' \
    --dataset your_corpus \
    --template default \
    --finetuning_type full \
    --output_dir path_to_output_checkpoint_path \
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
    --max_source_length 2048 \
    --dataloader_num_workers 8 \
    --val_size 0.01 \
    --bf16 \
    --overwrite_output_dir
```

Users can adjust on this basis to initiate their own training. For more detailed configurations, it is recommended to obtain the complete parameter list through `python src/train_bash.py -h`.

## Supervised Fine-Tuning
If you collect a batch of QA data and want to align it for devopspal, you can execute `scripts/devops-model-sft.sh` to initiate an additional training to align the model on the collected model. The specific code is as follows:

```bash
set -v 

torchrun --nproc_per_node=8 --nnodes=$WORLD_SIZE --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR --node_rank=$RANK src/train_bash.py \
    --deepspeed conf/deepspeed_config.json \
    --stage sft \
    --model_name_or_path path_to_model \
    --do_train \
    --report_to 'tensorboard' \
    --dataset your_corpus \
    --template chatml \
    --finetuning_type full \
    --output_dir /mnt/llm/devopspal/model/trained \
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
    --max_source_length 2048 \
    --dataloader_num_workers 8 \
    --val_size 0.01 \
    --bf16 \
    --overwrite_output_dir
```

Users can adjust on this basis to initiate their own SFT. For more detailed configurations, it is recommended to obtain the complete parameter list through `python src/train_bash.py -h`.

## Quantilization
We will provide quantitative models of the DevOps-Model-Chat series. Of course, you can also quantify your own trained models through the following code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model
import torch

model_name = "path_of_your_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

quantizer = GPTQQuantizer(bits=4, dataset="c4", block_name_to_quantize = "model.decoder.layers", model_seqlen = 2048)
quantized_model = quantizer.quantize_model(model, tokenizer)

out_dir = 'save_path_of_your_quantized_model'
quantized_model.save_quantized(out_dir)
```


# Disclaimer
Due to the characteristics of language models, the content generated by the model may contain hallucinations or discriminatory remarks. Please use the content generated by the DevOps-Model family of models with caution.
If you want to use this model service publicly or commercially, please note that the service provider needs to bear the responsibility for the adverse effects or harmful remarks caused by it. The developer of this project does not assume any responsibility for any consequences caused by the use of this project (including but not limited to data, models, codes, etc.) ) resulting in harm or loss.


# Acknowledgments
This project refers to the following open source projects, and I would like to express my gratitude to the relevant projects and research and development personnel.
- [LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning)
- [QwenLM](https://github.com/QwenLM)
