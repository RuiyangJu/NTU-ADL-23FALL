python ./multiple-choice/test_mc.py \
  --model_name_or_path ./multiple-choice/model \
  --config_name ./multiple-choice/model \
  --tokenizer_name ./multiple-choice/model \
  --test_file ./data/test.json \
  --context_file ./data/context.json \
  --max_length 512 \
  --per_device_test_batch_size 4 \
  --output_dir ./multiple-choice/output