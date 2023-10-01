# Homework 1 for NTU ADL 2023 Fall
## Path
Make sure you are in the correct path, otherwise:
```
  cd HW1
```
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
  python ./multiple-choice/run_swag_no_trainer.py \
    --model_name_or_path bert-base-chinese \
    --train_file ./data/train.json \
    --validation_file ./data/valid.json \
    --context_file ./data/context.json \
    --max_len 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-5 \
    --weight_decay 1e-6 \
    --num_train_epochs 10 \
    --output_dir ./multiple-choice/model
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
