# ${1}: path to train.jsonl. (./data/train.jsonl)
# ${2}: path to public.jsonl. (./data/public.jsonl)
# ${3}: path to the output. (./model)

python run_summarization.py \
  --model_name_or_path google/mt5-small \
  --train_file $1 \
  --validation_file $2 \
  --output_dir $3 \
  --do_train \
  --do_eval \
  --adafactor \
  --predict_with_generate \
  --pad_to_max_length \
  --learning_rate 3e-4 \
  --max_source_length 1024 \
  --max_target_length 128 \
  --num_train_epochs 50 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --per_device_eval_batch_size 4 \
  --eval_accumulation_steps 16 \
  --text_column maintext \
  --summary_column title \
  --overwrite_output_dir \
  --evaluation_strategy epoch \
  --logging_strategy epoch \