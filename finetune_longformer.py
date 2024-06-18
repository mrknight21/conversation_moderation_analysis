import os
import pandas as pd
import torch
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset, load_metric, Dataset
from utilities.metrics import compute_classification_accuracy, create_rogue_matric
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LongformerForSequenceClassification,
    LEDForSequenceClassification,
    LEDForConditionalGeneration
)

from attributes_prompts import dialogue_acts
import wandb

# Disable W&B logging
wandb.init(mode="disabled")

# Define the allowed choices
all_attributes = ["dialogue act", "social motive", "informational motive", "coordinative motive", "target speaker"]
models = ["allenai/longformer-base-4096", "allenai/longformer-large-4096", "allenai/led-base-16384", "MingZhong/DialogLED-large-5120"]

def parse_args():
    parser = argparse.ArgumentParser(description="QA perturbation experiment")

    parser.add_argument(
        "--data_path",
        default="./data/",
        type=str
    )

    parser.add_argument(
        "--corpus",
        choices=['insq', 'roundtable'],
        default='insq',
        type=str
    )

    parser.add_argument(
        "--output_dir",
        default="./output/",
        type=str
    )

    parser.add_argument(
        '--method',
        choices=['classification', 'sequence2sequence'],
        default='classification',
        help='Type of task to perform. Options are: classification, sequence2sequence. Default is classification.'
    )

    # Add an argument that accepts a list of strings with constrained options
    parser.add_argument(
        '--attributes',
        choices=all_attributes,
        nargs='+',
        default=all_attributes,
        help='List of aspects to train/evaluate. Options are: "dialogue act", "social motive", "informational motive", "coordinative motive", "target speaker".'
    )

    parser.add_argument(
        '--mode',
        choices=['train', 'eval'],
        default='train',
        help='selecting training mode or evaluation mode.'
    )

    parser.add_argument(
        '--model',
        choices=models,
        default=models[0],
        help='selecting training mode or evaluation model to use.'
    )

    parser.add_argument(
        '--epoch',
        type=int,
        default=3,
        help='epoch number for training'
    )

    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='batch size'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        help='checkpoint string for the mdoel'
    )

    args = parser.parse_args()
    return args

def generate_input_sequence(sample):
    input_string = f"Topic: {sample['topic']} \n"
    input_string += f"Speakers: {sample['speakers']} \n\n"
    if len(sample["prior context"]) > 0:
        input_string += f"Prior context: {sample['prior context']} \n\n"
    if len(sample["post context"]) > 0:
        input_string += f"Post context: {sample['prior context']} \n\n"
    input_string += f"Target: {sample['target']}"
    return input_string


def generate_output_sequence(sample):
    output_string = ""
    output_string += f"informational motive: {sample['informational motive']} \n"
    output_string += f"social motive: {sample['social motive']} \n"
    output_string += f"coordinative motive: {sample['coordinative motive']} \n"
    output_string += f"dialogue act: {sample['dialogue act']} \n"
    output_string += f"target speaker: {sample['target speaker']} \n"
    return output_string


def load_json_data(path, split = "train", source='gpt', method="classification"):
    data_set = []
    with open(path) as f:
        json_objs = json.load(f)
        for i in json_objs:
            sample = {
                "id": i["id"],
                "topic": i["topic"],
                "speakers": ",\n".join(i["speakers"]),
                "prior context": "".join([f"{s[0]} ({s[1]}): {s[2]} \n" for s in i["context"]["prior_context"]]),
                "post context": "".join([f"{s[0]} ({s[1]}): {s[2]} \n" for s in i["context"]["post_context"]]),
                "target": f"{i['target']['speaker']} ({i['target']['role']}): {i['target']['content']} \n"
            }
            if split == "train":
                sample["informational motive"] = "informational motive" in i["answer"]["motives"]
                sample["social motive"] = "social motive" in i["answer"]["motives"]
                sample["coordinative motive"] = "coordinative motive" in i["answer"]["motives"]
                sample["dialogue act"] = int(dialogue_acts[i["answer"]["dialogue act"]])
                sample["target speaker"] = int(i["answer"]['target speaker(s)'][0])
            else:
                if source == "gpt":
                    sample["informational motive"] = "informational motive" in i["answer"]["gpt"]["motives"]
                    sample["social motive"] = "social motive" in i["answer"]["gpt"]["motives"]
                    sample["coordinative motive"] = "coordinative motive" in i["answer"]["gpt"]["motives"]
                    sample["dialogue act"] = i["answer"]["gpt"]["dialogue act"]
                    sample["target speaker"] = i["answer"]["gpt"]['target speaker(s)']
            sample["input_sequence"] = generate_input_sequence(sample)
            sample["output_sequence"] = generate_output_sequence(sample)
            data_set.append(sample)
    return Dataset.from_list(data_set)

def main():
    args = parse_args()
    data_path = args.data_path

    model = args.model
    mode = args.mode
    method = args.method
    corpus = args.corpus
    batch_size = args.batch
    attributes = args.attributes



    if mode == "train":
    # load data
        train_data = load_json_data(data_path + corpus + "/agg/train.json", method=method)
        dev_data = load_json_data(data_path + corpus + "/agg/dev.json", split="dev", method=method)
    else:
        test_data = load_json_data(data_path + corpus + "agg/test.json", split="test", method=method)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)

    # max encoder length is 8192 for PubMed
    encoder_max_length = 3072
    decoder_max_length = 64

    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        inputs = tokenizer(
            batch["input_sequence"],
            padding="max_length",
            truncation=True,
            max_length=encoder_max_length,
        )

        if method != "classification":
            outputs = tokenizer(
                batch["output_sequence"],
                padding="max_length",
                truncation=True,
                max_length=decoder_max_length,
            )
            batch["labels"] = outputs.input_ids

            # We have to make sure that the PAD token is ignored
            batch["labels"] = [
                [-100 if token == tokenizer.pad_token_id else token for token in labels]
                for labels in batch["labels"]
            ]
        else:
            batch["labels"] = batch[attributes[0]]


        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask

        # create 0 global_attention_mask lists
        batch["global_attention_mask"] = len(batch["input_ids"]) * [
            [0 for _ in range(len(batch["input_ids"][0]))]
        ]

        # since above lists are references, the following line changes the 0 index for all samples
        batch["global_attention_mask"][0][0] = 1

        return batch


    # map train data
    train_data = train_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=['id', 'topic', 'speakers', 'prior context', 'post context', 'target', 'informational motive', 'social motive', 'coordinative motive', 'dialogue act', 'target speaker', 'input_sequence', 'output_sequence'],
    )

    # map val data
    dev_data = dev_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=['id', 'topic', 'speakers', 'prior context', 'post context', 'target', 'informational motive', 'social motive', 'coordinative motive', 'dialogue act', 'target speaker', 'input_sequence', 'output_sequence'],
    )

    # set Python list to PyTorch tensor
    train_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )

    # set Python list to PyTorch tensor
    dev_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )

    output_path = args.output_dir + corpus + "/" + model + "/"
    if not os.path.isdir(output_path):
        if not os.path.isdir(args.output_dir + corpus + "/" ):
            os.mkdir(args.output_dir + corpus + "/" )
        os.mkdir(output_path)

    # enable fp16 apex training
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # fp16=True,
        # fp16_backend="apex",
        output_dir=args.output_dir + corpus + "/",
        logging_steps=20,
        eval_steps=700,
        save_steps=300,
        warmup_steps=100,
        save_total_limit=2,
        gradient_accumulation_steps=2,
    )


    # compute Rouge score during validation


    print(f"loading model......{args.model}")
    # load model + enable gradient checkpointing & disable cache for checkpointing
    # led = AutoModelForSeq2SeqLM.from_pretrained(args.model_string, gradient_checkpointing=True, use_cache=False)
    if method != "classification":
        if args.checkpoint:
            longformer_model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
        else:
            longformer_model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

        # set generate hyperparameters
        longformer_model.config.num_beams = 4
        longformer_model.config.max_length = 512
        longformer_model.config.min_length = 100
        longformer_model.config.length_penalty = 2.0
        longformer_model.config.early_stopping = True
        longformer_model.config.no_repeat_ngram_size = 3
    else:
        if args.checkpoint:
            if "led" in model.lower():
                longformer_model = LEDForSequenceClassification.from_pretrained(args.checkpoint)
            else:
                longformer_model = LongformerForSequenceClassification.from_pretrained(args.checkpoint)
        else:
            if "led" in model.lower():
                longformer_model = LEDForSequenceClassification.from_pretrained(model)
            else:
                longformer_model = LongformerForSequenceClassification.from_pretrained(model)

    if method != "classification":
        compute_metrics = create_rogue_matric(tokenizer)
    else:
        compute_metrics = compute_classification_accuracy

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=longformer_model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=dev_data,
    )

    print("Start training......")

    # start training
    trainer.train()

def evaluate_longformer(checkpoint):
    # load rouge
    rouge = load_metric("rouge")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    led = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    # set generate hyperparameters
    led.config.num_beams = 2
    led.config.max_length = 144
    led.config.min_length = 20
    # led.config.length_penalty = 2.0
    led.config.early_stopping = True
    # led.config.no_repeat_ngram_size = 3

    insq_dev = load_json_data("./data/insq/agg/dev.json", split="dev")

    # max encoder length is 8192 for PubMed
    encoder_max_length = 3072
    decoder_max_length = 64
    batch_size = 2

    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        inputs = tokenizer(
            batch["input_sequence"],
            padding="max_length",
            truncation=True,
            max_length=encoder_max_length,
        )
        outputs = tokenizer(
            batch["output_sequence"],
            padding="max_length",
            truncation=True,
            max_length=decoder_max_length,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask

        # create 0 global_attention_mask lists
        batch["global_attention_mask"] = len(batch["input_ids"]) * [
            [0 for _ in range(len(batch["input_ids"][0]))]
        ]

        # since above lists are references, the following line changes the 0 index for all samples
        batch["global_attention_mask"][0][0] = 1
        batch["labels"] = outputs.input_ids

        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]

        return batch

    # map val data
    insq_dev = insq_dev.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=['id', 'topic', 'speakers', 'prior context', 'post context', 'target', 'informational motive', 'social motive', 'coordinative motive', 'dialogue act', 'target speaker', 'input_sequence', 'output_sequence'],
    )

    # set Python list to PyTorch tensor
    insq_dev.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )

    # enable fp16 apex training
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        resume_from_checkpoint=checkpoint,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # fp16=True,
        # fp16_backend="apex",
        output_dir="./",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # warmup_steps=100,
        # save_total_limit=2,
        gradient_accumulation_steps=2,
    )

    # compute Rouge score during validation
    def compute_metrics_rogue(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(
            predictions=pred_str, references=label_str, rouge_types=["rouge2"]
        )["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=led,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics_rogue,
        # train_dataset=insq_train,
        eval_dataset=insq_dev,
    )

    trainer.evaluate()








if __name__ == "__main__":
    main()

