python ./question-answering/train_qa.py \
    --model_name_or_path ./question-answering/model \
    --config_name ./question-answering/model \
    --tokenizer_name ./question-answering/model \
    --test_file ./data/test.json \
    --context_file ./data/context.json \
    --predict_mc_file ./multiple-choice/output/predict.json \
    --preprocessing_num_workers 12 \
    --max_length 512 \
    --doc_stride 128 \
    --do_predict \
    --per_device_eval_batch_size 1 \
    --output_dir ./question-answering/output