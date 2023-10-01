python ./multiple-choice/train_mc.py \
  --model_name_or_path bert-base-chinese \
  --train_file ./data/train.json \
  --validation_file ./data/valid.json \
  --context_file ./data/context.json \
  --max_length 512 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --learning_rate 3e-5 \
  --weight_decay 1e-6 \
  --num_train_epochs 10 \
  --output_dir ./multiple-choice/model