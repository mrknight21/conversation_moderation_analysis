import os
import pandas as pd
import torch
import numpy as np
import argparse
import json
from collections import OrderedDict
from tqdm import tqdm
from datasets import load_dataset, load_metric, Dataset
from utilities.metrics import compute_classification_accuracy, compute_classification_eval_report, create_multitask_classification_eval_metric, create_rogue_matric, compute_led_classification_eval_report
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LongformerForSequenceClassification,
    LEDForSequenceClassification,
    LEDConfig,
    LongformerConfig,
)
import wandb

from attributes_prompts import dialogue_acts
from meta_info import attributes_info
from longformer.multitasks_longformer import LongformerForSequenceMultiTasksClassification


# Disable W&B logging
wandb.init(mode="disabled")

# Define the allowed choices
all_attributes = ["dialogue_act", "social_motive", "informational_motive", "coordinative_motive", "target_speaker"]
all_models = ["allenai/longformer-base-4096", "allenai/longformer-large-4096", "allenai/led-base-16384", "MingZhong/DialogLED-large-5120"]
all_corpus = ['insq', 'roundtable']
all_tasks = ['classification', 'sequence2sequence', 'multi-tasks']



def parse_args():
    parser = argparse.ArgumentParser(description="QA perturbation experiment")

    parser.add_argument(
        "--data_path",
        default="./data/",
        type=str
    )

    parser.add_argument(
        "--corpus",
        choices=all_corpus,
        default=all_corpus[0],
        type=str
    )

    parser.add_argument(
        "--output_dir",
        default="./output/",
        type=str
    )

    parser.add_argument(
        '--method',
        choices=all_tasks,
        default=all_tasks[0],
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
        choices=['train', 'eval', "debug"],
        default='train',
        help='selecting training mode or evaluation mode.'
    )

    parser.add_argument(
        '--model',
        choices=all_models,
        default=all_models[0],
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

    input_string += f"Target: {sample['target']} \n\n"

    if len(sample["post context"]) > 0:
        input_string += f"Post context: {sample['post context']} \n\n"

    return input_string


def generate_output_sequence(sample):
    output_string = ""
    output_string += f"informational motive: {sample['informational_motive']} \n"
    output_string += f"social motive: {sample['social_motive']} \n"
    output_string += f"coordinative motive: {sample['coordinative_motive']} \n"
    output_string += f"dialogue act: {sample['dialogue_act']} \n"
    output_string += f"target speaker: {sample['target_speaker']} \n"
    return output_string


def load_json_data(path, split = "train", source='human', limit= -1):
    data_set = []
    with open(path) as f:
        json_objs = json.load(f)
        for num, i in enumerate(json_objs):
            sample = {
                "id": i["id"],
                "topic": i["topic"],
                "speakers": ",\n".join(i["speakers"]),
                "prior context": "".join([f"{s[0]} ({s[1]}): {s[2]} \n" for s in i["context"]["prior_context"]]),
                "post context": "".join([f"{s[0]} ({s[1]}): {s[2]} \n" for s in i["context"]["post_context"]]),
                "target": f"{i['target']['speaker']} ({i['target']['role']}): {i['target']['content']} \n"
            }
            if split == "train":
                sample["informational_motive"] = 1 if "informational motive" in i["answer"]["motives"] else 0
                sample["social_motive"] = 1 if "social motive" in i["answer"]["motives"] else 0
                sample["coordinative_motive"] = 1 if "coordinative motive" in i["answer"]["motives"] else 0
                sample["dialogue_act"] = int(dialogue_acts[i["answer"]["dialogue act"]])
                sample["target_speaker"] = int(i["answer"]['target speaker(s)'][0])
            else:
                if source == "gpt":
                    sample["informational_motive"] = 1 if "informational motive" in i["answer"]["gpt"]["motives"] else 0
                    sample["social_motive"] = 1 if "social motive" in i["answer"]["gpt"]["motives"] else 0
                    sample["coordinative_motive"] = 1 if "coordinative motive" in i["answer"]["gpt"]["motives"] else 0
                    sample["dialogue_act"] = i["answer"]["gpt"]["dialogue act"]
                    sample["target_speaker"] = i["answer"]["gpt"]['target speaker(s)']
                else:
                    sample["informational_motive"] = int(i["answer"]["human"]["motives"]["informational motive"]["label"])
                    sample["social_motive"] = int(i["answer"]["human"]["motives"]["social motive"]["label"])
                    sample["coordinative_motive"] = int(i["answer"]["human"]["motives"]["coordinative motive"]["label"])
                    sample["dialogue_act"] = int(i["answer"]["human"]["dialogue act"]["label"])
                    sample["target_speaker"] = int(i["answer"]["human"]["target speaker(s)"]["label"])
            sample["input_sequence"] = generate_input_sequence(sample)
            sample["output_sequence"] = generate_output_sequence(sample)
            data_set.append(sample)
            if -1 < limit <= num:
                break
    return Dataset.from_list(data_set)

def finetune_longformer():
    args = parse_args()
    data_path = args.data_path

    model = args.model
    mode = args.mode
    method = args.method
    corpus = args.corpus
    batch_size = args.batch
    attributes = args.attributes
    target_attributes_info = {attributes[0]: attributes_info[attributes[0]]}
    attribute_string = attributes[0]

    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        checkpoint = None

    if method == "multi-tasks" or "led" in model.lower():
        target_attributes_info = OrderedDict()
        attributes_logits_index_ranges = []
        attribute_index = 0
        for a in attributes:
            target_attributes_info[a] = attributes_info[a]
            num_labels = attributes_info[a]['num_labels']
            if len(attributes_logits_index_ranges) == 0:
                logits_range = (0, num_labels)
            else:
                last_index = attributes_logits_index_ranges[-1][-1]
                logits_range = (last_index, last_index + num_labels)
            target_attributes_info[a]["logits_range"] = logits_range
            target_attributes_info[a]["attribute_index"] = attribute_index
            attribute_index += 1
            attributes_logits_index_ranges.append(logits_range)

        if len(attributes) == len(all_attributes):
            attribute_string = "all"
        elif len(attributes) == 1:
            attribute_string = attributes[0]
        else:
            attribute_string = "_".join([ "".join([w[0] for w in a.split("_")]) for a in attributes])

    limit = -1
    if mode == "debug":
        limit = 10
        batch_size = 2

    if mode == "train" or mode == "debug":
    # load data
        train_data = load_json_data(data_path + corpus + "/agg/train.json", limit=limit)
        dev_data = load_json_data(data_path + corpus + "/agg/dev.json", split="dev", limit=limit)
    else:
        train_data = None
        dev_data = load_json_data(data_path + corpus + "/agg/test.json", split="test")

    # load tokenizer
    if checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model)

    # max encoder length is 8192 for PubMed
    if mode == "debug":
        encoder_max_length = 1024
    else:
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

        if method == "sequence2sequence":
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
        elif method == "multi-tasks" or "led" in model.lower():
            labels = []
            for att in attributes:
                labels.append(batch[att])
            labels = np.array(labels).T
            batch["labels"] = labels
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

    remove_columns = ['id', 'topic', 'speakers', 'prior context', 'post context', 'target', 'informational_motive', 'social_motive', 'coordinative_motive', 'dialogue_act', 'target_speaker', 'input_sequence', 'output_sequence']
    tensor_columns = ["input_ids", "attention_mask", "global_attention_mask", "labels"]

    if mode == "train" or mode == "debug":
        # map train data
        train_data = train_data.map(
            process_data_to_model_inputs,
            batched=True,
            batch_size=batch_size,
            remove_columns=remove_columns
        )

        # map val data
        dev_data = dev_data.map(
            process_data_to_model_inputs,
            batched=True,
            batch_size=batch_size,
            remove_columns=remove_columns
        )

        # set Python list to PyTorch tensor
        train_data.set_format(
            type="torch",
            columns=tensor_columns,
        )

        # set Python list to PyTorch tensor
        dev_data.set_format(
            type="torch",
            columns=tensor_columns,
        )
    else:
        # map test data
        dev_data = dev_data.map(
            process_data_to_model_inputs,
            batched=True,
            batch_size=batch_size,
            remove_columns=remove_columns
        )

        # set Python list to PyTorch tensor
        dev_data.set_format(
            type="torch",
            columns=tensor_columns,
        )

    output_path = args.output_dir + corpus + "/" + model.replace("/", "_") + "_" +attribute_string + "/"
    if not os.path.isdir(output_path):
        if not os.path.isdir(args.output_dir + corpus + "/" ):
            os.mkdir(args.output_dir + corpus + "/" )
        os.mkdir(output_path)

    if method == "sequence2sequence":
        # enable fp16 apex training
        training_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            evaluation_strategy="steps",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            # fp16=True,
            # fp16_backend="apex",
            output_dir=output_path,
            logging_steps=20,
            eval_steps=700,
            save_steps=300,
            warmup_steps=100,
            save_total_limit=2,
            gradient_accumulation_steps=4,
        )
    else:
        training_args = TrainingArguments(
            output_dir=output_path,
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=args.epoch,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )


        # compute Rouge score during validation


    print(f"loading model......{args.model}")




    if method == "sequence2sequence":
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
    elif method == "multi-tasks":
        if args.checkpoint:
            longformer_model = LongformerForSequenceMultiTasksClassification.from_pretrained(checkpoint)
        else:
            if "led" in model.lower():
                config = LEDConfig.from_pretrained(model)
            else:
                config = LongformerConfig.from_pretrained(model)
            config.longformer_encoder_base = model
            config.tasks_info = target_attributes_info
            longformer_model = LongformerForSequenceMultiTasksClassification(config)
    else:
        if checkpoint:
            if "led" in model.lower():
                longformer_model = LongformerForSequenceMultiTasksClassification.from_pretrained(checkpoint)
            else:
                longformer_model = LongformerForSequenceClassification.from_pretrained(args.checkpoint)
        else:
            if "led" in model.lower():
                config = LEDConfig.from_pretrained(model)
                config.longformer_encoder_base = model
                config.tasks_info = target_attributes_info
                longformer_model = LongformerForSequenceMultiTasksClassification(config)
            else:
                longformer_model = LongformerForSequenceClassification.from_pretrained(model, num_labels=target_attributes_info[attributes[0]]["num_labels"])

    if method == "sequence2sequence":
        compute_metrics = create_rogue_matric(tokenizer)
    elif method == "multi-tasks" or "led" in model.lower():
        compute_metrics = create_multitask_classification_eval_metric(target_attributes_info)
    else:
        compute_metrics = compute_classification_eval_report

    if method == "sequence2sequence":

        # instantiate trainer
        trainer = Seq2SeqTrainer(
            model=longformer_model,
            tokenizer=tokenizer,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_data,
            eval_dataset=dev_data,
        )

    else:

        trainer = Trainer(
            model=longformer_model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=dev_data,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
    if mode == "debug" or mode == "eval":
        print("Start evaluation......")
        eval_report = trainer.evaluate()
        if checkpoint:
            with open(f"./results/{corpus}/{'_'.join(checkpoint.split('/')[-2:]).replace('-', '_')}_eval.json", mode="w") as f:
                json.dump(eval_report, f)
        else:
            with open(f"./results/{corpus}/{model.replace('/', '_')}_{attributes[0]}_eval.json", mode="w") as f:
                json.dump(eval_report, f)


    if mode == "debug" or mode == "train":
        print("Start training......")
        # start training
        trainer.train()
