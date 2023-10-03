import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, Dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils_qa import postprocess_qa_predictions

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = get_logger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def save_prefixed_metrics(results, output_dir, file_name: str = "all_results.json", metric_key_prefix: str = "eval"):
    for key in list(results.keys()):
        if not key.startswith(f"{metric_key_prefix}_"):
            results[f"{metric_key_prefix}_{key}"] = results.pop(key)

    with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(results, f, indent=4)

def main():
    args = parse_args()
    send_example_telemetry("test_qa", args)
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    output_file = args.output_dir
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            args.output_dir = os.path.split(args.output_dir)[0]
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.test_file is not None:
            data_files["test"] = args.test_file
        extension = args.test_file.split(".")[-1]
        raw_datasets = {}
        with open(data_files["test"], 'r', encoding='utf-8') as f:
            raw_datasets['test'] = json.load(f)
            f.close()

    with open(args.context_file, 'r', encoding='utf-8') as f:
        context = json.load(f)
        f.close()
    print('context size:\t', len(context))

    with open(args.predict_mc_file, 'r', encoding='utf-8') as f:
        mc_pred = json.load(f)
        f.close()
    print('pred size:\t', len(mc_pred))

    for a in ['test']:
        tmp_dict = raw_datasets[a]
        tmp = [{'id': item['id'],
             'question': item['question'],
             'context': context[mc_pred[item['id']]],} 
             for item in tmp_dict]
        raw_datasets[a] = Dataset.from_list(tmp)

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForQuestionAnswering.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForQuestionAnswering.from_config(config)

    column_names = raw_datasets["test"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]
    pad_on_right = tokenizer.padding_side == "right"

    if args.max_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_length passed ({args.max_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_length={tokenizer.model_max_length}."
        )

    max_length = min(args.max_length, tokenizer.model_max_length)

    def prepare_train_features(examples):
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
        return tokenized_examples

    def prepare_validation_features(examples):
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    if args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_examples = raw_datasets["test"]
        if args.max_predict_samples is not None:
            test_examples = test_examples.select(range(args.max_predict_samples))
        with accelerator.main_process_first():
            predict_dataset = test_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
            if args.max_predict_samples is not None:
                predict_dataset = predict_dataset.select(range(args.max_predict_samples))

    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    if args.do_predict:
        predict_dataset_for_model = predict_dataset.remove_columns(["example_id", "offset_mapping"])
        predict_dataloader = DataLoader(predict_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    def post_processing_function(examples, features, predictions, stage="eval"):
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=args.version_2_with_negative,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            null_score_diff_threshold=args.null_score_diff_threshold,
            output_dir=args.output_dir,
            prefix=stage,
        )
        if args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        step = 0
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        for i, output_logit in enumerate(start_or_end_logits):
            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]
            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]
            step += batch_size
        return logits_concat

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    model, predict_dataloader = accelerator.prepare(model, predict_dataloader)

    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("qa_no_trainer", experiment_config)

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(predict_dataloader)
            resume_step -= starting_epoch * len(predict_dataloader)

    if args.do_predict:
        logger.info("***** Running Prediction *****")
        logger.info(f"  Num examples = {len(predict_dataset)}")
        logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

        all_start_logits = []
        all_end_logits = []

        model.eval()

        for step, batch in enumerate(predict_dataloader):
            print(f'Evaluating........  {step}/{len(predict_dataloader)}', end='\r')
            with torch.no_grad():
                outputs = model(**batch)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                if not args.pad_to_max_length: 
                    start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                    end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

                all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
                all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

        max_len = max([x.shape[1] for x in all_start_logits])
        start_logits_concat = create_and_fill_np_array(all_start_logits, predict_dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, predict_dataset, max_len)

        del all_start_logits
        del all_end_logits

        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(test_examples, predict_dataset, outputs_numpy)
        print(prediction.predictions[:10])

        with open(output_file, 'w') as f:
            f.write('id,answer\n')
            for item in prediction.predictions:
                f.write(item['id'] + ',"' + item['prediction_text'] + '"\n')
            f.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
    parser.add_argument("--dataset_name", type=str, default=None, help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).")
    parser.add_argument("--do_predict", action="store_true", help="To do prediction on the question answering model")
    parser.add_argument("--train_file", type=str, default=None, help="A csv or a json file containing the training data.")
    parser.add_argument("--validation_file", type=str, default=None, help="A csv or a json file containing the validation data.")
    parser.add_argument("--test_file", type=str, default=None, help="A csv or a json file containing the testing data.")
    parser.add_argument("--predict_mc_file", type=str, default=None, help="A csv or a json file containing the prediction data.")
    parser.add_argument("--context_file", type=str, default=None, help="A csv or a json file containing the context data.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=1, help="The number of workers to preprocess.")
    parser.add_argument("--max_length", type=int, default=128, help=("The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded if `--pad_to_max_lengh` is passed."))
    parser.add_argument("--pad_to_max_length", action="store_true", help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", required=False)
    parser.add_argument("--config_name", type=str, default=None, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--doc_stride", type=int, default=128, help="When splitting up a long document into chunks how much stride to take between chunks.")
    parser.add_argument("--n_best_size", type=int, default=20, help="The total number of n-best predictions to generate when looking for an answer.")
    parser.add_argument("--null_score_diff_threshold", type=float, default=0.0, help=("The threshold used to select the null answer: if the best answer has a score that is less than the score of the null answer minus this threshold, the null answer is selected for this example. Only useful when `version_2_with_negative=True`."))
    parser.add_argument("--version_2_with_negative", action="store_true", help="If true, some of the examples do not have an answer.")
    parser.add_argument("--max_answer_length", type=int, default=30,help=("The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another."))
    parser.add_argument("--max_train_samples", type=int, default=None, help=("For debugging purposes or quicker training, truncate the number of training examples to this value if set."))
    parser.add_argument("--max_eval_samples", type=int, default=None, help=("For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."))
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--max_predict_samples", type=int, default=None, help="For debugging purposes or quicker training, truncate the number of prediction examples to this")
    parser.add_argument("--model_type", type=str, default=None, help="Model type to use if training from scratch.", choices=MODEL_TYPES)
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--checkpointing_steps", type=str,default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="If the training should continue from a checkpoint folder.")
    parser.add_argument("--with_tracking", action="store_true", help="Whether to enable experiment trackers for logging.")
    parser.add_argument("--report_to", type=str, default="all", help=('The integration to report the results and logs to. Supported platforms are `"tensorboard"`, `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations. Only applicable when `--with_tracking` is passed.'))
    args = parser.parse_args()

    if (
        args.dataset_name is None
        and args.test_file is None
    ):
        raise ValueError("Need either a dataset name or a training/validation/test file.")
    else:
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

if __name__ == "__main__":
    main()
