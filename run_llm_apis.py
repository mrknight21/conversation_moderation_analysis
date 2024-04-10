import re
import openai
import random
import pandas as pd
import json
import time
import logging
import argparse
from tqdm import tqdm
from typing import List
import torch
import transformers
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

from feature_generation import generate_features

# random seed
random.seed(2023)

# openai settings
budget_org = ''
openai.organization = budget_org
openai.api_key = ''

model_list = ['gpt35', 'gpt4', 'llama2-7b', 'llama2-70b', 'led']
# maximum trials for model to get correct number of answers
MAX_TRIALS = 3

role_dict = {"for": "argue for", "against": "argue against", "mod": "moderator", "unknown":"audience"}

def extract_responses(input_string, keyword):
    # Define a regular expression pattern to extract responses based on the provided keyword
    pattern = re.compile(rf'{keyword} \d+:\s*(.*?)(?=\s*{keyword} \d+:|$)', re.I | re.S)
    matches = pattern.findall(input_string)
    return [match.strip() for match in matches]

def check_single_answer(output_text):

    try:
        answer = json.loads(output_text)
        if any([k not in answer for k in ["speaker name", "speaker role", "response"] ]):
            return False
        else:
            return True
    except Exception as e:
        return False
def gpt_batch(samples,batch_prompts, model_name='gpt35', is_single_unit=True):
    results = []
    api_func = None
    output = None
    if model_name == 'gpt35':
        api_func = gpt35
    elif model_name == 'gpt4':
        api_func = gpt4
    for i, sample in tqdm(enumerate(batch_prompts)):
        trial = 0
        is_ans_validate = False
        while trial < MAX_TRIALS:
            output = gpt_single(sample, api_func=api_func, tries=3, wait_time=1)
            if check_single_answer(output):
                is_ans_validate = True
                break
            else:
                pass
            trial += 1
        if is_ans_validate:
            results.append(json.loads(output))
        else:
            results.append('Answer not correct.')
    return results

def gpt_single(input_text, api_func, tries=3, wait_time=1):
    output_text = None
    for n in range(tries + 1):
        if n == tries:
            raise openai.error.ServiceUnavailableError(f"Tried {tries} times.")
        try:
            output_text = api_func(input_text)
        except (openai.error.ServiceUnavailableError, openai.error.APIError, openai.error.APIConnectionError,
                openai.error.RateLimitError, openai.error.Timeout) as e:
            logging.warning(e)
            logging.warning(f"Retry after {wait_time}s. (Trail: {n + 1})")
            time.sleep(wait_time)
            continue
        break
    return output_text

def gpt35(input_text, model="gpt-3.5-turbo-16k"):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{'role': 'user', 'content': input_text}],
        temperature=1
    )
    output_text = completion.choices[0].message.content
    return output_text


def gpt4(input_text, model="gpt-4"):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{'role': 'user', 'content': input_text}],
        max_tokens=25,
        temperature=1
    )
    output_text = completion.choices[0].message.content
    return output_text


def llama2(samples, batch_prompts, model_name="llama2-7b", is_single_unit=True):
    if model_name == "llama2-7b":
        model_path = "meta-llama/Llama-2-7b-chat-hf"
        torch_dtype = torch.float32
    else:
        model_path = "meta-llama/Llama-2-70b-chat-hf"
        torch_dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=4096, padding_side="right",
                                              use_fast=False)
    results = []
    for i,sample in tqdm(enumerate(batch_prompts)):
        # assert if input_text is with token limitation
        tokenized_text = tokenizer.encode(sample)
        if len(tokenized_text) > tokenizer.model_max_length:
            raise AssertionError(
                f"Input length exceed limit (4096), your input length is {len(tokenized_text)} tokens.")

        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        sequences = pipeline(
            sample,
            do_sample=True,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=1000,
            temperature=1,
            return_full_text=False
        )
        output_text = sequences[0]['generated_text']
        trial = 0
        is_ans_correct = False
        while trial < MAX_TRIALS:
            if check_single_answer(samples[i], output_text, is_single_unit):
                is_ans_correct = True
                break
            trial += 1
        if is_ans_correct:
            results.append(output_text)
        else:
            results.append('Answer not correct.')
    return results

def get_prompt_unit(instance, llama_format=False):
    topic = f"Dialogue topic: {instance.title}\n\n"
    speakers = "Speakers: \n" + instance.speakers + "\n\n"
    dialogue_history = "Dialogue history: \n" + instance.dialogue_history + "\n\n"
    context = topic + speakers + dialogue_history

    instruction = "In your capacity as the moderator of this debate, featuring teams arguing 'for' and 'against' a given topic, your task is to monitor the dialogue closely. Based on the subject matter of the debate, detailed information about the speakers, and the progression of the dialogue as provided, decide whether an intervention from you is necessary. If you choose to intervene, specify precisely what you would say to steer the conversation in a constructive direction or to address any issues that have arisen. If you believe no intervention is necessary, predict who will take the floor next, specifying their role in the debate ('for', 'against', or 'mod') and what they are likely to say next.\n"
    answer_format = 'Your decision and response must be concise and only presented in the following JSON format: {"speaker name": String, "speaker role": String, "response": String} \n'
    constraint = "Note: The 'response' field should contain only a single sentence of the predicted spoken content. This may be your intervention as the moderator or your prediction of the next contribution to the debate. \n"
    instruction = instruction + answer_format + constraint
    if llama_format:
        prompt = '<s>[INST]' + '\n\n'.join(['<<SYS>>' + instruction + answer_format + '<</SYS>>', context]) + '\n[/INST]'
    else:
        prompt = '\n\n'.join([instruction, context])
    return prompt

def led(samples, batch_prompts, model_name="/data/scratch/projects/punim0478/ruixing/models/checkpoint-210", is_single_unit=True):
    # hyperparameters
    min_predict_length = 1
    max_predict_length = 512
    do_sample = True
    top_p = 0.95
    num_beams = 5
    no_repeat_ngram_size = 3
    length_penalty = 0.6
    results = []
    # load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True).to("cuda")
    for i,sample in tqdm(enumerate(batch_prompts)):
        # assert if input_text is with token limitation
        tokenized_text = tokenizer.encode(sample)
        if len(tokenized_text) > tokenizer.model_max_length:
            raise AssertionError(
                f"Input length exceed limit{tokenizer.model_max_length}, your input length is {len(tokenized_text)} tokens.")
        input_dict = tokenizer(
            sample,
            padding=True,
            return_tensors="pt",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        input_ids = input_dict.input_ids.to("cuda")
        attention_mask = input_dict.attention_mask.to("cuda")
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[:, 0] = 1
        global_attention_mask[input_ids == tokenizer.bos_token_id] = 1

        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            max_length=max_predict_length,
            min_length=min_predict_length,
            do_sample=do_sample,
            top_p=top_p,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty
        )
        generation = tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)

        output_text = generation
        trial = 0
        is_ans_correct = False
        while trial < MAX_TRIALS:
            if check_single_answer(samples[i], output_text, is_single_unit):
                is_ans_correct = True
                break
            trial += 1
        if is_ans_correct:
            results.extend(generation)
        else:
            results.append('Answer not correct.')
    return results

def run_model(samples: List, model_name: str = 'gpt35'):
    # check if model_name is valid
    if model_name not in model_list:
        raise AssertionError("Model name not valid, please choose from gpt35, gpt4, llama2-7b, llama2-70b, led.")
    if model_name.startswith('llama'):
        llama_format = True
    else:
        llama_format = False
    # run model
    # 1. check input format and build prompt from inputs
    batch_prompts: List = []

    print("synthesize prompts from inputs using Scenario 0,1 and 2 setting...")
    for i, sample in tqdm(samples.iterrows()):
        batch_prompts.append(get_prompt_unit(sample, llama_format=llama_format))
    print("synthesize prompts finished.")

    # 2. run corresponding models and return results
    print(f"run model {model_name}...")
    if model_name == 'gpt35' or model_name == 'gpt4':
        results = gpt_batch(samples, batch_prompts, model_name=model_name)
    elif model_name.startswith('llama'):
        results = llama2(samples, batch_prompts, model_name=model_name)
    elif model_name == 'led':
        results = led(samples, batch_prompts)
    else:
        raise AssertionError(
            "Model name not valid, please choose from gpt35, gpt4, llama2-7b, llama2-70b, led.")
    return results, batch_prompts

def evaluate_result(data, results, prompts):
    role_dict = {
        "mod": 0,
        "for": 1,
        "against": 2,
        "unknown": -1
    }

    data["prompt"] = prompts
    data["pred_response"] = [r["response"] for r in results]
    data["pred_speaker"] = [r["speaker name"] for r in results]
    data["pred_role"] = [r["speaker role"] for r in results]
    data["pred_intervene"] = [r["speaker role"] == "mod" for r in results]
    data["gt_intervene"] = data["ans_role"] == "mod"

    report = {
        "speaker accuracy": (data["pred_speaker"] == data["ans_speaker"]).sum() / len(results),
        "role accuracy": (data["pred_role"] == data["ans_role"]).sum() / len(results),
        "moderation accuracy": (data["pred_intervene"] == data["gt_intervene"]).sum() / len(results),
    }
    return report, data


def parse_args():
    parser = argparse.ArgumentParser(description="QA perturbation experiment")
    parser.add_argument(
        "-model_name",
        type=str,
        default="gpt35",
        choices=model_list,
        help="The model to run.",
    )
    parser.add_argument(
        "--context_utt_num",
        default=10,
        type=int
    )

    parser.add_argument(
        "--max_input_length",
        default=2048,
        type=int
    )

    parser.add_argument(
        "--max_dialogue_num",
        default=10,
        type=int
    )

    parser.add_argument(
        "--sample_per_dialogue",
        default=30,
        type=int
    )

    parser.add_argument(
        "--role",
        default=None,
        type=str
    )

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


def main():
    args = parse_args()
    model_name = args.model_name
    test_data = pd.read_csv(args.data_path + "isd_sampled_test.csv")
    # train_data = pd.read_csv(args.data_path + "isd_train.csv")

    test_results, test_prompts = run_model(test_data)
    report, output_table = evaluate_result(test_data[:len(test_results)], test_results, test_prompts)
    output_table.to_csv("./" + model_name + "_output.csv")
    pass


if __name__ == "__main__":
    main()
    pass