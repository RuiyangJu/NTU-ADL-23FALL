python ./question-answering/train_qa.py \
  --model_name_or_path hfl/chinese-lert-large \
  --train_file ./data/train.json \
  --validation_file ./data/valid.json \
  --context_file ./data/context.json \
  --max_length 512 \
  --with_tracking \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --output_dir ./question-answering/model
