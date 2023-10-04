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

### Test
```
  bash test_qa.sh
```
