import os
import sys
import nltk
import logging
import numpy as np
import jsonlines
import datasets
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional
from filelock import FileLock
import transformers
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_offline_mode, send_example_telemetry
from trainer_summarization import Seq2SeqTrainer
from transformers import (
    set_seed,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainingArguments
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError("Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files")
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50TokenizerFast]

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".jsonl"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    send_example_telemetry("run_summarization", model_args, data_args)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training/evaluation parameters {training_args}")
    if data_args.source_prefix is None and model_args.model_name_or_path in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
        logger.warning("You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with `--source_prefix 'summarize: ' `")
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")
    set_seed(training_args.seed)
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir, use_auth_token=True if model_args.use_auth_token else None)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
            if extension == "jsonl":
                extension = "json"
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
            if extension == "jsonl":
                extension = "json"
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
            if extension == "jsonl":
                extension = "json"
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir, use_auth_token=True if model_args.use_auth_token else None)
    config = AutoConfig.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path, cache_dir=model_args.cache_dir, revision=model_args.model_revision, use_auth_token=True if model_args.use_auth_token else None)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer, revision=model_args.model_revision, use_auth_token=True if model_args.use_auth_token else None)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, from_tf=bool(".ckpt" in model_args.model_name_or_path), config=config, cache_dir=model_args.cache_dir, revision=model_args.model_revision, use_auth_token=True if model_args.use_auth_token else None)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    if (hasattr(model.config, "max_position_embeddings") and model.config.max_position_embeddings < data_args.max_source_length):
        if model_args.resize_position_embeddings is None:
            logger.warning(f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} to {data_args.max_source_length}.")
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings} position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the model's position encodings by passing `--resize_position_embeddings`.")
    prefix = data_args.source_prefix if data_args.source_prefix is not None else "None"
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return
    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (data_args.lang is not None), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"
        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang
        forced_bos_token_id = (tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None)
        model.config.forced_bos_token_id = forced_bos_token_id
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}")
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}")
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False
    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(f"label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for `{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory")

    def preprocess_function(examples):
        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache, desc="Running tokenizer on train dataset")

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(preprocess_function, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache, desc="Running tokenizer on validation dataset")

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        ids = predict_dataset["id"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(preprocess_function, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache, desc="Running tokenizer on prediction dataset")

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8 if training_args.fp16 else None)

    gen_kwargs = {"do_sample": data_args.do_sample, "top_k": data_args.top_k, "top_p": data_args.top_p, "temperature": data_args.temperature}
    trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=train_dataset if training_args.do_train else None,eval_dataset=eval_dataset if training_args.do_eval else None, tokenizer=tokenizer, data_collator=data_collator, compute_metrics=None)
    
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        max_train_samples = (data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset))
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    results = {}
    max_length = (training_args.generation_max_length if training_args.generation_max_length is not None else data_args.val_max_target_length)
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval", gen_kwargs=gen_kwargs)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams, gen_kwargs=gen_kwargs)
        metrics = predict_results.metrics
        max_predict_samples = (data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset))
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = data_args.output_file if data_args.output_file is not None else os.path.join(training_args.output_dir, f"{prefix}_predictions.json")
                with jsonlines.open(output_prediction_file, 'w') as writer:
                    for id, title in zip(ids, predictions):
                        writer.write({"title": title, "id": id})
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name
    if data_args.lang is not None:
        kwargs["language"] = data_args.lang
    return results

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
}

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."})
    model_revision: str = field(default="main", metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."})
    use_auth_token: bool = field(default=False, metadata={"help": ("Will use the token generated when running `huggingface-cli login` (necessary to use this script with private models).")})
    resize_position_embeddings: Optional[bool] = field(default=None, metadata={"help": ("Whether to automatically resize the position embeddings if `max_source_length` exceeds the model's position embeddings.")})

@dataclass
class DataTrainingArguments:
    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."})
    text_column: Optional[str] = field(default=None, metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."})
    summary_column: Optional[str] = field(default=None, metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."})
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."})
    validation_file: Optional[str] = field(default=None, metadata={"help": ("An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file).")})
    test_file: Optional[str] = field(default=None, metadata={"help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."})
    output_file: Optional[str] = field(default=None, metadata={"help": "An optional output file to predict test file" "(a jsonlines or csv file)."})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "The number of processes to use for the preprocessing."})
    max_source_length: Optional[int] = field(default=256, metadata={"help": ("The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")})
    max_target_length: Optional[int] = field(default=64, metadata={"help": ("The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")})
    val_max_target_length: Optional[int] = field(default=None, metadata={"help": ("The maximum total sequence length for validation target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`. This argument is also used to override the ``max_length`` param of ``model.generate``, which is used during ``evaluate`` and ``predict``.")})
    pad_to_max_length: bool = field(default=False, metadata={"help": ("Whether to pad all samples to model maximum sentence length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More efficient on GPU but very bad for TPU.")})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": ("For debugging purposes or quicker training, truncate the number of training examples to this value if set.")})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": ("For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.")})
    max_predict_samples: Optional[int] = field(default=None, metadata={"help": ("For debugging purposes or quicker training, truncate the number of prediction examples to this value if set.")})
    num_beams: Optional[int] = field(default=None, metadata={"help": ("Number of beams to use for evaluation. This argument will be passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.")})
    do_sample: Optional[bool] = field(default=False, metadata={"help": "Whether or not to use sampling ; use greedy decoding otherwise. This argument will be passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."})
    top_k: Optional[int] = field(default=None, metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k-filtering. This argument will be passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."})
    top_p: Optional[float] = field(default=None, metadata={"help": "If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation. This argument will be passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."})
    temperature: Optional[float] = field(default=None, metadata={"help": "The value used to module the next token probabilities. This argument will be passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."})
    ignore_pad_token_for_loss: bool = field(default=True, metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."})
    source_prefix: Optional[str] = field(default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."})
    forced_bos_token: Optional[str] = field(default=None, metadata={"help": ("The token to force as the first generated token after the decoder_start_token_id. Useful for multilingual models like mBART where the first generated token needs to be the target language token (Usually it is the target language token)")})

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.test_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

if __name__ == "__main__":
    main()
