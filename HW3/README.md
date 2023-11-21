# Homework 3 for NTU ADL 2023 Fall
## Envirement Preparation
```
  pip install -r requirements.txt
```
If you want to enable the quantized LoRA (QLoRA) on the Windows platform, you will be required to install a pre-built version of bitsandbytes library, which supports CUDA 11.1 to 12.1.
```
  pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.39.1-py3-none-win_amd64.whl
```
## Data Processing
### Train Data
The provided train.json format is as follows:
```
[
  {
    "id": "db63fb72-e211-4596-94a4-69617706f7ef",
    "instruction": "翻譯成文言文：\n雅裏惱怒地說： 從前在福山田獵時，你誣陷獵官，現在又說這種話。\n答案：",
    "output": "雅裏怒曰： 昔畋於福山，卿誣獵官，今復有此言。"
  },
  ...
]
```
To conform to the format of the [alpaca_zh](https://huggingface.co/datasets/shibing624/alpaca-zh) dataset, as follows:
```
[
  {
    "instruction": "保持健康的三个提示。",
    "input": "",
    "output": "以下是保持健康的三个提示：\n\n1. 保持身体活动。每天做适当的身体运动，如散步、跑步或游泳，能促进心血管健康，增强肌肉力量，并有助于减少体重。\n\n2. 均衡饮食。每天食用新鲜的蔬菜、水果、全谷物和脂肪含量低的蛋白质食物，避免高糖、高脂肪和加工食品，以保持健康的饮食习惯。\n\n3. 睡眠充足。睡眠对人体健康至关重要，成年人每天应保证 7-8 小时的睡眠。良好的睡眠有助于减轻压力，促进身体恢复，并提高注意力和记忆力。"
  },
  ...
]
```
We run [train_process.py](train_process.py) to process the data as follows:
```
[
  {
    "instruction": "翻譯成文言文：\n雅裏惱怒地說： 從前在福山田獵時，你誣陷獵官，現在又說這種話。\n答案：",
    "input": "",
    "output": "雅裏怒曰： 昔畋於福山，卿誣獵官，今復有此言。"
  },
  ...
]
```
### Public Test Data
For public_test.json, we run [public_process.py](public_process.py) to process the data format as follows:
```
[
  {
    "instruction": "翻譯成文言文：\n於是，廢帝讓瀋慶之的堂侄、直將軍瀋攸之賜瀋慶之毒藥，命瀋慶之自殺。",
    "input": "",
    "output": "帝乃使慶之從父兄子直閣將軍攸之賜慶之藥。"
  },
  ...
]
```
### Private Test Data
For private_test.json, we run [private_process.py](private_process.py) to process the data format as follows:
```
[
  {
    "instruction": "穿右穴而進，其下甚削，陷峽頗深，即下穿所入之峽也，以壁削路阻，不得達。\n幫我把這句話翻譯成現代文",
    "input": "",
    "output": ""
  },
  ...
]
```
### Prediction Data
In order to conform to the requirement for prediction in the Homework (DO NOT include any special tokens (<s>, </s>, …) and your prompt in your output), we run [result_process.py](result_process.py) to process the data format as follows:
```
[
  {
    "id": "d573ddd1-7bb9-468d-b906-e392223d9579",
    "output": "穿過右穴之後，下去竟然很深，下峽頗深，就是下穿進去之峽，邊壁崩塌，阻止不通。"
  },
  ...
]
```

## Download
Use gdown to download trained models, tokenizers and data from Google Drive:
```
  bash download.sh
```

## Train
```
  bash train.sh /path/to/Taiwan-LLaMa-folder
```
Before `bash train.sh`, you need to put `Taiwan-LLM-7B-v2.0-chat` into this folder.
For example:
```
  bash train.sh ./
```

#### Hyperparameters:
| Quantization_Bit | FlashAttention-2 | Max_Tokens | Learning_Rate | Max_Samples | Maximum_Gradient_Norm | Num_Epochs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 4 | True | 1024 | 1e-5 | 10000 | 1.0 | 10 |

| Compute_Type | Learning_Rate_Scheduler | Lora_Rank | Lora_Dropout | Lora_Target | Batch_size | Gradient_Accumulation |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| bf16 | cosine | 8 | 0.1 | q_proj, v_proj | 4 | 16 |

## Evaluation
Model Performance on public_test.json:
```
  python ppl.py --base_model_path Taiwan-LLM-7B-v2.0-chat --peft_path model --test_data_path data/public_test.json
```
Output:
```
  Mean perplexity: 3.712290671825409
```

## Run
Before `bash run.sh`, you need to put `Taiwan-LLM-7B-v2.0-chat` into the `./model` folder.
```
  bash run.sh /path/to/model-folder /path/to/input.josn /path/to/output.json
```
For example:
```
  bash run.sh ./model ./data/private_test.json ./prediction.json
```
