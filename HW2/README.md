# Homework 2 for NTU ADL 2023 Fall
## Envirement Preparation
```
  pip install -r requirements.txt
  pip install -e tw_rouge
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
  bash run.sh ./data/public.jsonl ./data/output.jsonl
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
| model | max_source_len | max_target_len | pad_to_max_len | learning_rate | batch_size | num_epochs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| google/mt5-small | 1024 | 128 | True | 3e-4 | 64 | 50 |

### Validation
| Strategies | Rouge-1 | Rouge-2 | Rouge-l |
| :---: | :---: | :---: | :---: |
| beams=4 | 28.17 | 11.07 | 24.86 |
| beams=16 | 28.25 | 11.14 | 24.92 |

### Evaluation
```
usage: eval.py [-h] [-r REFERENCE] [-s SUBMISSION]

optional arguments:
  -h, --help            show this help message and exit
  -r REFERENCE, --reference REFERENCE
  -s SUBMISSION, --submission SUBMISSION
```
For example:
```
  python eval.py -r public.jsonl -s submission.jsonl
```
