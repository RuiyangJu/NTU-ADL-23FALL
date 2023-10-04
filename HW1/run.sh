# ${1}: path to context.json. (./data)
# ${2}: path to test.json. (./data)
# ${3}: path to the output prediction file named prediction.csv. (./question-answering/output/output.csv)

python ./multiple-choice/test_mc.py \
  --model_name_or_path ./multiple-choice/model \
  --config_name ./multiple-choice/model \
  --tokenizer_name ./multiple-choice/model \
  --test_file ${2} \
  --context_file ${1} \
  --max_length 512 \
  --per_device_test_batch_size 4 \
  --output_dir ./multiple-choice/output

python ./question-answering/test_qa.py \
  --model_name_or_path ./question-answering/model \
  --config_name ./question-answering/model \
  --tokenizer_name ./question-answering/model \
  --test_file ${2} \
  --context_file ${1} \
  --predict_mc_file ./multiple-choice/output/predict.json \
  --preprocessing_num_workers 12 \
  --max_length 512 \
  --doc_stride 128 \
  --do_predict \
  --per_device_eval_batch_size 1 \
  --output_dir ${3}
