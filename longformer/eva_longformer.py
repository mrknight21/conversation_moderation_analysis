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

device = "mps"
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
    tokenizer = AutoTokenizer.from_pretrained('results/checkpoint-500')
    # load model + enable gradient checkpointing & disable cache for checkpointing
    model = AutoModelForSequenceClassification.from_pretrained('results/checkpoint-500', num_labels=2).to("mps")

    test_data = pre_process_dataset(pd.read_csv("data/archived/isd_sampled_test.csv"), tokenizer)
    test_data = Dataset.from_pandas(test_data, split="test")



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
        batch["global_attention_mask"] = [0 for _ in range(len(batch["input_ids"]))]



        # since above lists are references, the following line changes the 0 index for all samples
        batch["global_attention_mask"][0] = 1

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
        remove_columns=["input", "output_text", "output_role", "output_moderation"]
    )


    # set Python list to PyTorch tensor
    test_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"]
    )

    def eval_performance(instance):
        for k, v in instance.items():
            instance[k] = instance[k].to("mps")
        output = model(**instance)
        return output

    eval_results = test_data.map(eval_performance)







if __name__ == "__main__":
    main()