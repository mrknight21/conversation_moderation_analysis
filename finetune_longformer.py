import os
import pandas as pd
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset, load_metric, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, LongformerForSequenceClassification
from transformers import DataCollatorWithPadding


DATA_FOLDER = '.'
TRAIN_FOLDER = os.path.join(DATA_FOLDER, 'train')
TEST_FOLDER = os.path.join(DATA_FOLDER, 'test')
os.environ["WANDB_DISABLED"] = "true"


max_input_length = 8192
max_output_length = 512
batch_size = 2

role_dict = {
        "mod": 0,
        "for": 1,
        "against": 2
    }

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

def pre_process_dataset(dataset, tokenizer):

    processed_dataset = []
    for i, r in dataset.sample(frac = 1).iterrows():
        input = "Title: " + r.title + "." + "\n"
        input += r.speakers + "." + "\n"
        input += r.dialogue_history

        output = r.ans_speaker + f"<{r.ans_role}>:" +r.ans_text
        role_label = role_dict[r.ans_role]
        moderator_intervene_label = int(r.ans_role == "mod")

        processed_dataset.append({"input":input, "output_text": output, "output_role": role_label, "output_moderation": moderator_intervene_label})
    df = pd.DataFrame(processed_dataset)
    return df


def main():
    args = parse_args()
    model_list = ['allenai/longformer-base-4096', "allenai/led-base-16384"]

    # load rouge
    rouge = load_metric("rouge")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')

    test_data = pre_process_dataset(pd.read_csv("./data/isd_sampled_test.csv"), tokenizer)
    test_data = Dataset.from_pandas(test_data, split="test")
    train_sampled_data = pre_process_dataset(pd.read_csv("./data/isd_sampled_test.csv"), tokenizer)
    train_data = Dataset.from_pandas(train_sampled_data, split="train")



    # max encoder length is 8192 for PubMed
    encoder_max_length = 1024
    decoder_max_length = 256
    batch_size = 2

    def process_data_to_model_inputs(batch, task="binary_class"):
        # tokenize the inputs and labels
        inputs = tokenizer(
            batch["input"],
            padding="max_length",
            truncation=True,
            max_length=encoder_max_length,
        )
        outputs = tokenizer(
            batch["output_text"],
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

        if task == "generation":
            batch["labels"] = outputs.input_ids

            # We have to make sure that the PAD token is ignored
            batch["labels"] = [
                [-100 if token == tokenizer.pad_token_id else token for token in labels]
                for labels in batch["labels"]
            ]
        elif task == "multi_class":
            batch["labels"] = batch["output_roles"]
        elif task == "binary_class":
            batch["labels"] = batch["output_moderation"]
        return batch

    # map val data
    test_data = test_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["input", "output_text", "output_role", "output_moderation"]
    )

    # map train data
    train_data = train_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["input", "output_text", "output_role", "output_moderation"]
    )


    # set Python list to PyTorch tensor
    test_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"]
    )

    # set Python list to PyTorch tensor
    train_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"]
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

    # load model + enable gradient checkpointing & disable cache for checkpointing
    model = AutoModelForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels = 2).to("mps")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        use_mps_device=True
    )

    trainer = Trainer(

        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # start training
    trainer.train()



if __name__ == "__main__":
    main()
    pass