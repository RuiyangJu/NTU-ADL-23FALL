# Homework 1 for NTU ADL 2023 Fall
## Envirement Preparation
```
  pip install -r requirements.txt
```
If there are still some packages that have not been installed successfully, you need to install them manually. Examples are as follows:
```
  pip install datasets
  pip install evaluate
  pip install accelerate
```

## Download
Use gdown to download trained models, tokenizers and data from Google Drive:
```
  bash download.sh
```

## Run
```
bash run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv
```
For example:
```
bash run.sh ./data/context.json ./dataset/test.json ./data/output/prediction.csv
```

## Paragraph Selection (multiple choice)
### Train
```
  bash train_mc.sh
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

### Test
```
  bash test_mc.sh
```

## Span Selection (question answering)
### Train
```
  bash train_qa.sh
```

#### Hyperparameters:
| model | max_len | batch_size | gradient_accmulation_steps | learning_rate | num_epochs |
| :---: | :---: | :---: | :---: | :---: | :---: |
| hfl/chinese-lert-large | 512 | 4 | 16 | 3e-5 | 10 |

### Validation
| model | num_epoch | exact_match | train_loss |
| :---: | :---: | :---: | :---: |
| hfl/chinese-lert-large | 1 | 83.616 | 0.980 |
| hfl/chinese-lert-large | 2 | 83.383 | 0.514 |
| hfl/chinese-lert-large | 3 | 84.280 | 0.368 |
| hfl/chinese-lert-large | 4 | 84.513 | 0.301 |
| hfl/chinese-lert-large | 5 | 83.981 | 0.277 |
| hfl/chinese-lert-large | 6 | 84.114 | 0.242 |
| hfl/chinese-lert-large | 7 | 83.716 | 0.119 |
| hfl/chinese-lert-large | 8 | 84.546 | 0.084 |
| hfl/chinese-lert-large | 9 | 84.513 | 0.082 |
| hfl/chinese-lert-large | 10 | 83.948 | 0.072 |

### Test
```
  bash test_qa.sh
```
