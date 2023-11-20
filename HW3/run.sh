# ${1}: path to the folder containing the files downloaded by download.sh
# ${2}: path to the input file (.json)
# ${3}: path to the output file (.json)

python preprocess.py --input ${2}

python src/train_bash.py \
    --stage sft \
    --model_name_or_path ${1}/Taiwan-LLM-7B-v2.0-chat \
    --predict_with_generate True \
    --checkpoint_dir ${1}/checkpoint \
    --finetuning_type lora \
    --template default \
    --flash_attn True \
    --shift_attn False \
    --dataset_dir data \
    --dataset predict \
    --cutoff_len 1024 \
    --max_samples 10000 \
    --per_device_eval_batch_size 8 \
    --max_new_tokens 128 \
    --top_p 0.7 \
    --temperature 0.95 \
    --output_dir saved \
    --do_predict True

python postprocess.py --original ${2} --generated saved/generated_predictions.jsonl --output ${3}