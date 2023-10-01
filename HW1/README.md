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

## Paragraph Selection (multiple choice)
### Train
```
  bash train_mc.sh
```

#### Hyperparameters:
| model | max_len | batch_size | gradient_accmulation_steps | learning_rate | weight_decay | num_epochs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| bert-base-chinese | 512 | 4 | 16 | 3e-5 | 1e-6 | 10 |

### Test
```
  bash test_mc.sh
```

## Span Selection (question answering)
### Train
