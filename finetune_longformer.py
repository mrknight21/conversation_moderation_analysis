import os
import pandas as pd
import torch
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset, load_metric, Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from attributes_prompts import dialogue_acts

def parse_args():
    parser = argparse.ArgumentParser(description="QA perturbation experiment")

    parser.add_argument(
        "--data_path",
        default="./data/",
        type=str
    )

    parser.add_argument(
        "--output_dir",
        default="./output/",
        type=str
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


def load_json_data(path, split = "train", source='gpt'):
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
    model_list = ["allenai/led-base-16384", "MingZhong/DialogLED-large-5120"]

    # load rouge
    rouge = load_metric("rouge")

    # load data
    insq_train = load_json_data("./data/insq/agg/train.json")
    insq_dev = load_json_data("./data/insq/agg/dev.json", split="dev")


    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("MingZhong/DialogLED-large-5120")


    # max encoder length is 8192 for PubMed
    encoder_max_length = 3072
    decoder_max_length = 64
    batch_size = 16


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


    # map train data
    insq_train = insq_train.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=['id', 'topic', 'speakers', 'prior context', 'post context', 'target', 'informational motive', 'social motive', 'coordinative motive', 'dialogue act', 'target speaker', 'input_sequence', 'output_sequence'],
    )

    # map val data
    insq_dev = insq_dev.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=['id', 'topic', 'speakers', 'prior context', 'post context', 'target', 'informational motive', 'social motive', 'coordinative motive', 'dialogue act', 'target speaker', 'input_sequence', 'output_sequence'],
    )

    # set Python list to PyTorch tensor
    insq_train.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )

    # set Python list to PyTorch tensor
    insq_dev.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )

    # enable fp16 apex training
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # fp16=True,
        # fp16_backend="apex",
        output_dir="./",
        logging_steps=20,
        eval_steps=700,
        save_steps=300,
        warmup_steps=100,
        save_total_limit=2,
        gradient_accumulation_steps=2,
    )


    # compute Rouge score during validation
    def compute_metrics(pred):
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

    print("loading model......")
    # load model + enable gradient checkpointing & disable cache for checkpointing
    led = AutoModelForSeq2SeqLM.from_pretrained("MingZhong/DialogLED-large-5120", gradient_checkpointing=True, use_cache=False)

    # set generate hyperparameters
    led.config.num_beams = 4
    led.config.max_length = 512
    led.config.min_length = 100
    led.config.length_penalty = 2.0
    led.config.early_stopping = True
    led.config.no_repeat_ngram_size = 3


    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=led,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=insq_train,
        eval_dataset=insq_dev,
    )

    print("Start training......")

    # start training
    trainer.train()




if __name__ == "__main__":
    main()
    pass