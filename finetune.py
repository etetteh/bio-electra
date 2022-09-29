from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import shutil
import argparse
import warnings

import numpy as np

import torch
import evaluate
import transformers

from util import utils
from datasets import load_dataset, load_metric

from transformers import (
    ElectraTokenizerFast,
    ElectraForSequenceClassification,
    ElectraForTokenClassification,
        
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    TrainingArguments, 
    Trainer
)
 
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
torch.cuda.empty_cache()


def main(args, trial):
    os.environ['PYTHONHASHSEED']=str(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        
    transformers.logging.set_verbosity_info()
    
    # load tokenizer from pretrained model
    global tokenizer
    tokenizer = ElectraTokenizerFast.from_pretrained(
        args.model_ckpt,
        tokenize_chinese_chars=True,
        strip_accents=True,
        lowercase=args.lowercase,
        is_fast=True,
        verbosity=0
    )

    # load dataset
    loader = "./downstream/biore.py" if args.loader == "re" else "./downstream/bioner.py"
    raw_datasets = load_dataset(loader, args.dataset)
    
    # load relation extraction model and tokenize dataset
    if args.loader == "re":
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        tokenized_datasets = raw_datasets.map(re_tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        
        model = ElectraForSequenceClassification.from_pretrained(
                    args.model_ckpt, 
                    num_labels=tokenized_datasets["train"].features["labels"].num_classes
        )
    else:
        # load named-entity recognition model and tokenize dataset
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        
        tokenized_datasets = raw_datasets.map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
        )
        
        ner_feature = raw_datasets["train"].features["ner_tags"]
        label_names = ner_feature.feature.names
        id2label = {str(i): label for i, label in enumerate(label_names)}
        label2id = {v: k for k, v in id2label.items()}
        
        model = ElectraForTokenClassification.from_pretrained(
                    args.model_ckpt,
                    id2label=id2label,
                    label2id=label2id,
        )
        
    output_dir = f"{args.output_dir}_{args.dataset}/trial_{trial+1}"

    # training arguments
    training_args = TrainingArguments(
        seed=args.seed,
        data_seed=args.seed,
        do_train=args.do_train,
        do_eval=args.do_eval,

        overwrite_output_dir=args.overwrite,
        output_dir=output_dir, 
        evaluation_strategy=args.eval_strategy, 
        save_strategy=args.save_strategy, 
        save_steps=args.save_steps,

        greater_is_better=args.greater_is_better, 
        load_best_model_at_end=args.load_best, 
        gradient_checkpointing=args.gradient_checkpointing,
        metric_for_best_model=args.metric, 

        label_smoothing_factor=args.smoothing,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.scheduler,
        optim="adamw_torch",

        dataloader_drop_last=False,
        auto_find_batch_size=True,                      
        
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        
        fp16=args.fp16,
    )

    # trainer    
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=re_compute_metrics if args.loader == "re" else ner_compute_metrics
    )
    
    # start model training
    train_result = trainer.train()

    # save model and log train metrics
    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(tokenized_datasets["train"])
    trainer.save_model()
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    # log evaluation metrics
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(tokenized_datasets["validation"])
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # perform inference and log test metrics
    predictions, labels, test_metrics = trainer.predict(tokenized_datasets["test"])
    trainer.log_metrics("predict", test_metrics)
    trainer.save_metrics("predict", test_metrics)


def re_tokenize_function(example):
    """
    A function to tokenize dataset batches.
    Args: 
        example: dataset batch 
    Returns: 
        tokenized dataset batch
    """
    return tokenizer(example["sentence1"], truncation=True, max_length=args.max_len)


def re_compute_metrics(eval_preds):
    """
    This function computes metrics 
    for Relation Extraction task.
    
    Args:
        eval_preds: model prediction
    Returns:
        accuracy, precision, recall, f1
    """
    metric = load_metric("./downstream/remetrics.py")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    average = "binary"
    if args.dataset != "gad": 
        average = "micro"
    return metric.compute(predictions = predictions, references = labels, average = average)


def ner_compute_metrics(eval_preds):
    """
    This function computes metrics 
    for Named-Entity Recognition task.
    
    Args:
        eval_preds: model prediction
    Returns:
        accuracy, precision, recall, f1
    """
    metric = evaluate.load("seqeval")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


def align_labels_with_tokens(labels, word_ids):
    """This function aligns samples and its labels"""
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(examples):
    """
    A function to tokenize dataset batches.
    Args: 
        example: dataset batch 
    Returns: 
        tokenized dataset batch
    """
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=128
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

    
def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="ELECTRA For Biomedical Data Fine-Tuning", add_help=add_help)
    
    parser.add_argument("--model_ckpt", "--ckpt", type=str, required=True, help="dir of pytorch checkpoint converted from original tensorflow checkpoint ")
    parser.add_argument("--lowercase", type=bool, default=True, help="Whether to use lowercase for tokenization")
    parser.add_argument("--max_len", type=int, default=128, help="max length of sequence")
    parser.add_argument("--loader", type=str, choices=["re", "ner"], required=True, help="finetuning dataset loader. re for Relation Extraction and ner for Named-Entity Recognition. Automatically downloads and processes the dataset")
    parser.add_argument("--dataset", type=str, required=True, help="")
    parser.add_argument("--output_dir", type=str, required=True, help="dir to save finetuning checkpoints, metrics, and logs")

    parser.add_argument("--do_train", type=bool, default=True, help="Train model")
    parser.add_argument("--do_eval", type=bool, default=True, help="Evaluate model")
    parser.add_argument("--trials", type=int, required=True, default=5, help="Number of trials. Each using a different seed")

    parser.add_argument("--overwrite", type=bool, default=True, help="Whether to overwrite existing output")

    parser.add_argument("--eval_strategy", type=str, choices=["epoch", "steps"], default="epoch", help="When to perform evaluation")
    parser.add_argument("--save_strategy", type=str, choices=["epoch", "steps"], default="epoch", help="When to save/log checkpoints")
    parser.add_argument("--save_steps", type=int, default=1000, help="Number of steps to run before logging")

    parser.add_argument("--greater_is_better", action="store_true", required=True, help="Whether metric is being minimized or maximized")
    parser.add_argument("--metric", type=str, choices=["precision", "recall", "f1"], default="f1", required=True, help="Which metric to optimize")
    parser.add_argument("--load_best", type=bool, default=True, help="Whether to load best model after training. Note the best model is needed for inference on test set")
    parser.add_argument("--fp16", action="store_true", help="")
    parser.add_argument("--bf16", action="store_true", help="")

    parser.add_argument("--smoothing", type=int, default=0.0, help="The smooth factor for labels")

    parser.add_argument("--learning_rate", "--lr", type=int, default=3e-4, help="learning rate or step size")
    parser.add_argument("--scheduler", 
                        "--sched", type=str, 
                        choices=["linear", "cosine", "polynomial", "cosine_with_restarts"], 
                        default="linear", help="The learning rate scheduler to use"
                       )
    parser.add_argument("--weight_decay", "--wd", type=float, default=0.0, help="weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="warmup ratio")

    parser.add_argument("--train_batch_size", "--tbs", type=int, default=8, help="train batch size")
    parser.add_argument("--eval_batch_size", "--ebs", type=int, default=8, help="evaluation batch size")
    parser.add_argument("--gradient_checkpointing", "--gckpt", type=bool, default=True, help="Whether to checkpoint the gradients.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    
    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir) 
    
    for trial in range(args.trials):
        heading_info = f"Model={args.model_ckpt}, Dataset={args.dataset}, Trial {trial+1}/{args.trials}"
        heading = lambda msg: utils.heading(msg + ": " + heading_info)
        heading("Started Training")
        
        args.seed = torch.initial_seed() % 2**32 * trial
        
        print("Config:")
        utils.log_config(args)
        
        main(args, trial)
        torch.cuda.empty_cache()
    
