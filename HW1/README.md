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
| model | max_len | batch_size | gradient_accmulation_steps | learning_rate | weight_decay | num_epochs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| bert-base-chinese | 512 | 4 | 16 | 3e-5 | 1e-6 | 10 |
### Test
```
  python ./multiple-choice/test_mc.py \
    --model_name_or_path ./multiple-choice/model/pytorch_model.bin \
    --config_name ./multiple-choice/model/config.json \
    --tokenizer_name ./multiple-choice/model \
    --test_file ./data/test.json \
    --context_file ./data/context.json \
    --max_length 512 \
    --per_device_test_batch_size 32 \
    --output_dir ./output/multiple-choice
```
