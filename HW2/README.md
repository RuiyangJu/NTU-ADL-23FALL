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
bash run.sh /path/to/train.jsonl /path/to/public.jsonl /path/to/output
```
For example:
```
bash run.sh ./data/train.jsonl ./data/public.jsonl ./model
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
| model | max_len | batch_size | gradient_accmulation_steps | learning_rate | num_epochs |
| :---: | :---: | :---: | :---: | :---: | :---: |
| hfl/chinese-macbert-large | 512 | 4 | 16 | 3e-5 | 2 |

### Validation
| model | num_epoch | accuracy |
| :---: | :---: | :---: |
| hfl/chinese-macbert-large | 1 | 0.967 |
| hfl/chinese-macbert-large | 2 | 0.964 |
