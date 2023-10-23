# ${1}: path to public.jsonl. (./data/public.jsonl)
# ${2}: path to output.jsonl. (./data/output.jsonl)

python run_summarization.py \
  --do_predict \
  --model_name_or_path ./model \
  --test_file $1\
  --output_file $2 \
  --predict_with_generate \
  --text_column maintext \
  --summary_column title \
  --per_device_eval_batch_size 4 \
  --num_beams 16 \