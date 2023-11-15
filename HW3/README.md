# Homework 3 for NTU ADL 2023 Fall
## Envirement Preparation
```
  pip install -r requirements.txt
```

## Data Processing
### Training Data
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




## Download
Use gdown to download trained models, tokenizers and data from Google Drive:
```
  bash download.sh
```

## Run
```
  bash run.sh /path/to/input.jsonl /path/to/output.jsonl
```
For example:
```
  bash run.sh ./data/public.jsonl ./data/submission.jsonl
```

## Evaluation
```
usage: eval.py [-h] [-r REFERENCE] [-s SUBMISSION]

optional arguments:
  -h, --help            show this help message and exit
  -r REFERENCE, --reference REFERENCE
  -s SUBMISSION, --submission SUBMISSION
```
For example:
```
  python eval.py -r ./data/public.jsonl -s ./data/submission.jsonl
```
Output:
```
{
  "rouge-1": {
    "r": 0.2514407046285254,
    "p": 0.2815113516763089,
    "f": 0.2585712911257173
  },
  "rouge-2": {
    "r": 0.09881004830495194,
    "p": 0.10841936828246301,
    "f": 0.10052484246314036
  },
  "rouge-l": {
    "r": 0.2234307264405613,
    "p": 0.24996033587187794,
    "f": 0.22957756251039388
  }
}
```

## Chinese News Summarization (Title Generation)
### Train
```
  bash train.sh /path/to/train.jsonl /path/to/public.jsonl /path/to/output
```
For example:
```
  bash train.sh ./data/train.jsonl ./data/public.jsonl ./model
```

#### Hyperparameters:
| model | max_source_len | max_target_len | pad_to_max_len | learning_rate | Optimizer | batch_size | num_epochs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| google/mt5-small | 1024 | 128 | True | 3e-4 | AdaFactor | 64 | 50 |

### Validation
| Strategies | Rouge-1 | Rouge-2 | Rouge-l |
| :---: | :---: | :---: | :---: |
| beams=4 | 25.77 | 10.00 | 22.92 |
| beams=16 | 25.86 | 10.05 | 22.96 |
| Top-k=10 | 22.71 | 7.90 | 20.07 |
| Top-p=0.9 | 21.63 | 7.49 | 19.18 |

