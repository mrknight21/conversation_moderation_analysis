import re

import numpy as np
from  openai import OpenAI
import random
import pandas as pd
import json
import time
import logging
import argparse
import glob
from tqdm import tqdm
from collections import Counter
from typing import List
from sklearn.metrics import classification_report
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

from attributes_prompts import construct_prompt_unit


# random seed
random.seed(2023)

mode = "dev"

INPUT_FOLDER = f"./data/{mode}_data"
OUTPUT_FOLDER = "./data/gpt_annotated_data"
EVA_FOLDER = "./data/evaluation_scores"


client = OpenAI(api_key='', organization='')

model_list = ['gpt35', 'gpt4', 'llama2-7b', 'llama2-70b', 'led']
da_encoder = {"probing": 0, "confronting": 1, "instruction": 2, "interpretation": 3, "supplement": 4,
              "all utility": 5}
# maximum trials for model to get correct number of answers
MAX_TRIALS = 3
PRIOR_CONTEXT_SIZE = 5
POST_CONTEXT_SIZE = 2

role_dict = {"for": "argue for", "against": "argue against", "mod": "moderator", "unknown":"audience"}

def extract_responses(input_string, keyword):
    # Define a regular expression pattern to extract responses based on the provided keyword
    pattern = re.compile(rf'{keyword} \d+:\s*(.*?)(?=\s*{keyword} \d+:|$)', re.I | re.S)
    matches = pattern.findall(input_string)
    return [match.strip() for match in matches]

def check_single_answer(output_text, instance):

    try:
        answer = json.loads(output_text)
        if any([k not in answer for k in ["motives", "dialogue act", "target speaker(s)"] ]):
            return False
        else:
            motives = answer["motives"]
            if motives:
                for m in motives:
                    if m not in ["informational motive", "social motive", "coordinative motive"]:
                        return False
            dialogue_act = answer["dialogue act"]
            if dialogue_act not in ["Probing", "Confronting", "Supplement", "Interpretation", "Instruction", "All Utility"]:
                return False
            speaker = answer["target speaker(s)"]
            if "speakers" in instance["meta"]:
                if speaker not in instance["meta"]["speakers"]:
                    return False
        return True

    except Exception as e:
        return False
def gpt_batch(batch_prompts, model_name='gpt35', is_single_unit=True):
    results = []
    api_func = None
    output = None
    if model_name == 'gpt35':
        api_func = gpt35
    elif model_name == 'gpt4':
        api_func = gpt4
    for i, instance in tqdm(enumerate(batch_prompts)):
        trial = 0
        is_ans_validate = False
        while trial < MAX_TRIALS:
            output = gpt_single(instance["prompt"], api_func=api_func, tries=3, wait_time=1)
            if check_single_answer(output, instance):
                is_ans_validate = True
                break
            else:
                pass
            trial += 1
        if is_ans_validate:
            instance["output"] = json.loads(output)
        else:
            instance["output"] = None
        results.append(instance)
    return results

def gpt_single(input_text, api_func, tries=3, wait_time=1):
    output_text = None
    for n in range(tries + 1):
        if n == tries:
            raise OpenAI.error.ServiceUnavailableError(f"Tried {tries} times.")
        try:
            output_text = api_func(input_text)
        except Exception as e:
            logging.warning(e)
            logging.warning(f"Retry after {wait_time}s. (Trail: {n + 1})")
            time.sleep(wait_time)
            continue
        break
    return output_text

def gpt35(input_text, model="gpt-3.5-turbo-16k"):
    completion = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': input_text}],
        temperature=1
    )
    output_text = completion.choices[0].message.content
    return output_text


def gpt4(input_text, model="gpt-4o"):
    completion = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': input_text}],
        temperature=1,
        response_format={"type": "json_object"}
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


def parse_args():
    parser = argparse.ArgumentParser(description="QA perturbation experiment")
    parser.add_argument(
        "-model_name",
        type=str,
        default="gpt4",
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

def process_episode(episode, model_name):
    meta = get_episode_meta(episode)
    debate = pd.read_excel(episode, index_col=0)
    episode_eval_data = {
        "dialogue act":{"human":[], "llm":[], "votes":[]},
        "informational motive":{"human":[], "llm":[], "votes":[]},
        "social motive":{"human":[], "llm":[], "votes":[]},
        "coordinative motive":{"human":[], "llm":[], "votes":[]},
        "target speaker":{"human":[], "llm":[], "votes":[]}
    }

    episode_eval_result = {'topic': meta['topic']}

    task_instances = []
    for i, r in debate.iterrows():
        if r.role == "mod":
            utt_id = int(r.id.split("_")[0])
            sent_id = int(r.id.split("_")[1])
            prior_context_mask = debate.id.apply(lambda x: not (
                    not (utt_id - PRIOR_CONTEXT_SIZE <= int(x.split("_")[0]) < utt_id) and not (int(x.split("_")[0]) == utt_id and \
                                                                                                int(x.split("_")[1]) < sent_id)))
            post_context_mask = debate.id.apply(lambda x: not (
                    not (utt_id + POST_CONTEXT_SIZE >= int(x.split("_")[0]) > utt_id) and not (int(x.split("_")[0]) == utt_id and \
                                                                                                int(x.split("_")[1]) > sent_id)))
            context = {
                "prior_context":[],
                "post_context": []
            }

            if len(debate[prior_context_mask]) > 0:
                context["prior_context"] = [(v.speaker, v.role, v.text) for i, v in debate[prior_context_mask].iterrows()]

            if len(debate[post_context_mask ]) > 0:
                context["post_context"] = [(v.speaker, v.role, v.text) for i, v in debate[post_context_mask].iterrows()]
            instance = {
                "id": r.id,
                "meta": meta,
                "context": context,
                "target": (r.speaker, r.role, r.text)
            }
            prompt = construct_prompt_unit(instance)
            instance["prompt"] = prompt
            task_instances.append(instance)

    results = gpt_batch(task_instances, model_name=model_name)
    ids = debate.id.tolist()
    human_im = debate["informational motive"].tolist()
    human_sm = debate["social motive"].tolist()
    human_cm = debate["coordinative motive"].tolist()
    human_da = debate["dialogue act"].tolist()
    human_sp = debate["target speaker"].tolist()

    human_im_vote = debate["informational motive vote"].tolist()
    human_sm_vote = debate["social motive vote"].tolist()
    human_cm_vote = debate["coordinative motive vote"].tolist()
    human_da_vote = debate["dialogue act vote"].tolist()
    human_sp_vote = debate["target speaker vote"].tolist()

    llm_im = [np.nan] * len(ids)
    llm_sm = [np.nan] * len(ids)
    llm_cm = [np.nan] * len(ids)
    llm_da = [np.nan] * len(ids)
    llm_sp = [np.nan] * len(ids)
    llm_rs = [np.nan] * len(ids)

    for r in results:
        try:
            id = r["id"]
            index = ids.index(id)
            llm_da[index] = r["output"]["dialogue act"]
            if not np.isnan(human_da[index]):
                episode_eval_data["dialogue act"]["llm"].append(da_encoder[r["output"]["dialogue act"].lower()])
                episode_eval_data["dialogue act"]["human"].append(human_da[index])
                episode_eval_data["dialogue act"]["votes"].append(get_votes(human_da_vote[index]))

            llm_sp[index] = r["output"]["target speaker(s)"]
            if not np.isnan(human_sp[index]):
                episode_eval_data["target speaker"]["llm"].append(meta['speakers'].index(r["output"]["target speaker(s)"]))
                episode_eval_data["target speaker"]["human"].append(human_sp[index])
                episode_eval_data["target speaker"]["votes"].append(get_votes(human_sp_vote[index]))

            llm_rs[index] = r["output"]["reason"]
            motives = r["output"]["motives"]

            llm_im[index] = 1 if "informational motive" in motives else 0
            if not np.isnan(human_im[index]):
                episode_eval_data["informational motive"]["llm"].append(llm_im[index])
                episode_eval_data["informational motive"]["human"].append(human_im[index])
                episode_eval_data["informational motive"]["votes"].append(get_votes(human_im_vote[index]))

            llm_sm[index] = 1 if "social motive" in motives else 0
            if not np.isnan(human_sm[index]):
                episode_eval_data["social motive"]["llm"].append(llm_sm[index])
                episode_eval_data["social motive"]["human"].append(human_sm[index])
                episode_eval_data["social motive"]["votes"].append(get_votes(human_sm_vote[index]))

            llm_cm[index] = 1 if "coordinative motive" in motives else 0
            if not np.isnan(human_cm[index]):
                episode_eval_data["coordinative motive"]["llm"].append(llm_cm[index])
                episode_eval_data["coordinative motive"]["human"].append(human_cm[index])
                episode_eval_data["coordinative motive"]["votes"].append(get_votes(human_cm_vote[index]))
        except Exception as e:
            print(f"topic {meta['topic']}, id: {id}")
            print(e)

    debate["informational motive llm"] = llm_im
    debate["social motive llm"] = llm_sm
    debate["coordinative motive llm"] = llm_cm
    debate["dialogue act llm"] = llm_da
    debate["target speaker llm"] = llm_sp
    debate["llm reason"] = llm_rs

    for k, v in episode_eval_data.items():
        episode_eval_result[k] = evaluate_human_llm_differences(v)

    output_path = OUTPUT_FOLDER + "/" + episode.split("/")[-1].replace(".xlsx", "") + f"_{mode}_llm.xlsx"
    debate.to_excel(output_path)
    return episode_eval_data, episode_eval_result

def get_votes(vote_string):
    vote_string = vote_string.replace("'", '"')
    votes = json.loads(vote_string)
    vote_counts = Counter(list(votes.values()))
    return vote_counts


def weighted_accuracy(human_vote_weights_dicts, llm):
    accuracy_scores = []
    for i, pred in enumerate(llm):
        votes_count = human_vote_weights_dicts[i]
        max_v = max(votes_count.values())
        score = votes_count.get(pred, 0) / max_v
        accuracy_scores.append(score)
    return np.mean(accuracy_scores)
def evaluate_human_llm_differences(eval_data):
    eval_report = classification_report(eval_data['human'], eval_data['llm'], output_dict=True)
    if eval_data["votes"]:
        eval_report["weighted_accuracy"] = weighted_accuracy(eval_data["votes"], eval_data["llm"])
    return eval_report

def get_episode_meta(episode):
    meta_json_path = episode.replace(".xlsx", "_meta.json")
    meta = None
    with open(meta_json_path) as f:
        metas = json.load(f)
        if len(metas.values()) > 0:
            meta = list(metas.values())[0]
        meta["topic"] = " ".join(meta["topic"].split("_")[2:])
    return meta

def main():
    args = parse_args()
    model_name = args.model_name

    episodes = glob.glob(INPUT_FOLDER + "/*.xlsx")
    evaluation_results = {}
    corpus_eval_data = {
        "dialogue act":{"human":[], "llm":[], "votes":[]},
        "informational motive":{"human":[], "llm":[], "votes":[]},
        "social motive":{"human":[], "llm":[], "votes":[]},
        "coordinative motive":{"human":[], "llm":[], "votes":[]},
        "target speaker":{"human":[], "llm":[], "votes":[]}
    }
    for e in episodes:
        if "~$" in e:
            continue
        topic = " ".join(e.split(".")[1].split("_")[3:])
        eps_data, eps_eval_report = process_episode(e, model_name)
        evaluation_results[topic] = eps_eval_report
        for k, v in corpus_eval_data.items():
            for t, r in v.items():
                r.extend(eps_data[k][t])
    overall_eval_report = {}
    for k, v in corpus_eval_data.items():
        overall_eval_report[k] = evaluate_human_llm_differences(v)
    evaluation_results["overall"] = overall_eval_report
    with open(EVA_FOLDER + f'/{mode}_llm_eva_scores.json', 'w') as fp:
        json.dump(evaluation_results, fp)



if __name__ == "__main__":
    main()