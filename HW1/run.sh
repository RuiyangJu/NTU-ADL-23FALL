python ./multiple-choice/test_mc.py \
  --model_name_or_path ./multiple-choice/model \
  --config_name ./multiple-choice/model \
  --tokenizer_name ./multiple-choice/model \
  --test_file ./data/test.json \
  --context_file ./data/context.json \
  --max_length 512 \
  --per_device_test_batch_size 4 \
  --output_dir ./multiple-choice/output

python ./question-answering/test_qa.py \
  --model_name_or_path ./question-answering/model_7 \
  --config_name ./question-answering/model_7 \
  --tokenizer_name ./question-answering/model_7 \
  --test_file ./data/test.json \
  --context_file ./data/context.json \
  --predict_mc_file ./multiple-choice/output/predict.json \
  --preprocessing_num_workers 12 \
  --max_length 512 \
  --doc_stride 128 \
  --do_predict \
  --per_device_eval_batch_size 1 \
  --output_dir ./question-answering/output.csv