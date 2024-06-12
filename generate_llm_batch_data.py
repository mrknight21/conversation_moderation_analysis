import glob
import pandas as pd
from openai import OpenAI
from run_llm_apis import get_episode_meta, gpt_batch
from attributes_prompts import construct_prompt_unit
import json
import re

MODE = "test"
INPUT_FOLDER = f"./data/{MODE}_data"
OUPUT_FOLDER = f"./data/batch_data/{MODE}"
LOG_FOLDER = "./log"
MODEL = "gpt-4o"
PRIOR_CONTEXT_SIZE = 5
POST_CONTEXT_SIZE = 2

# openai settings
# budget_org = 'org-IVogYSNZgs6F06KgvMk2UTnW'
# openai.organization = budget_org
# openai.api_key = 'sk-proj-ExP2VrJI9eiCgdrpJPqyT3BlbkFJzQxSKdlUslTBBizefj0C'
client = OpenAI(api_key='sk-proj-ExP2VrJI9eiCgdrpJPqyT3BlbkFJzQxSKdlUslTBBizefj0C', organization='org-IVogYSNZgs6F06KgvMk2UTnW')

def process_episode(episode, model_name):
    meta = get_episode_meta(episode)
    debate = pd.read_excel(episode, index_col=0)

    gpt_prompts = []
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
            gpt_call = {"custom_id": episode.split("/")[-1].replace(".xlsx", "") + "_" + r.id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {"model": model_name,
                                 "messages": [{"role": "user", "content": prompt}],
                                 "temperature": 1,
                                 "response_format": {"type": "json_object"},
                                 "max_tokens": 300}
                        }

            gpt_prompts.append(gpt_call)
    return gpt_prompts
def dicts_to_jsonl(data_list: list, filename: str) -> None:
    """
    Method saves list of dicts into jsonl file.
    :param data: (list) list of dicts to be stored,
    :param filename: (str) path to the output file. If suffix .jsonl is not given then methods appends
        .jsonl suffix into the file.
    :param compress: (bool) should file be compressed into a gzip archive?
    """
    sjsonl = '.jsonl'
    # Check filename
    if not filename.endswith(sjsonl):
        filename = filename + sjsonl
    # Save data
    with open(filename, 'w') as out:
        for ddict in data_list:
            jout = json.dumps(ddict) + '\n'
            out.write(jout)

def post_openai_batch_request(batch_data_file, part_index=-1):
    # upload batch request data
    batch_input_file = client.files.create(
        file=open(batch_data_file, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    response = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"{MODE} part {part_index}"
        }
    )

    print(f"batch file id: {batch_input_file_id}")
    print("client response: ")
    print(response)
    return batch_input_file_id, response.id

def main():

    episodes = sorted(glob.glob(INPUT_FOLDER + "/*.xlsx"))
    chunk_limit = 20
    batch_data = []
    part_index = 0
    batch_record = {}
    for i, e in enumerate(episodes):
        if "~$" in e:
            continue

        eps_data = process_episode(e, MODEL)
        batch_data.extend(eps_data)
        if (i + 1) % chunk_limit == 0:
            print(f"uploading batch part {part_index}!")
            batch_data_file = OUPUT_FOLDER + f"/isq_batch_{MODE}_part{part_index}.jsonl"
            dicts_to_jsonl(batch_data, batch_data_file)
            batch_input_file_id, batch_id = post_openai_batch_request(batch_data_file, part_index)
            batch_record[part_index] = {"file_id": batch_input_file_id, "batch_id": batch_id}
            part_index += 1
            batch_data = []
    if len(batch_data) != 0:
        print(f"uploading batch part {part_index}!")
        batch_data_file = OUPUT_FOLDER + f"/isq_batch_{MODE}_part{part_index}.jsonl"
        dicts_to_jsonl(batch_data, batch_data_file)
        batch_input_file_id, batch_id = post_openai_batch_request(batch_data_file, part_index)
        batch_record[part_index] = {"file_id": batch_input_file_id, "batch_id": batch_id}
        part_index += 1
        batch_data = []
    # Convert and write JSON object to file
    with open(LOG_FOLDER + "/batch_upload_record.json", "w") as outfile:
        json.dump(batch_record, outfile)

def check_status(file):
    with open(file) as f:
        records = json.load(f)
        if len(records) > 0:
            for index, ids in records.items():
                print(f"part {index} status:")
                batch_info = client.batches.retrieve(ids["batch_id"])
                print(batch_info.status)

def check_single_answer(output_text):
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
        return True
    except Exception as e:
        return False

def download_batch_output(batch_id, part_index=-1, process_invalid=False):
    print(f"{batch_id} status:")
    batch_info = client.batches.retrieve(batch_id)
    print(batch_info.status)
    json_outputs = []
    invalid_outputs = []
    if batch_info.status == "completed":
        content = client.files.content(batch_info.output_file_id)
        if part_index > -1:
            filename = OUPUT_FOLDER + f"/isq_batch_{MODEL}_{MODE}_output_part{part_index}.jsonl"
        else:
            filename = OUPUT_FOLDER + f"/isq_batch_{MODEL}_{MODE}_output.jsonl"
        with open(filename, 'w') as out:
            json_pattern = re.compile(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}')
            for l in content.iter_lines():
                out.write(l)
                json_output = json.loads(l)
                custom_id = json_output["custom_id"]
                response = json_output["response"]
                status_code = response["status_code"]
                if status_code != 200:
                    print(f"failed request at custom id: {custom_id}")
                else:
                    body = response["body"]
                    model_used = body["model"]
                    answer = body["choices"][0]["message"]["content"]
                    matches = json_pattern.findall(answer)
                    if len(matches) > 0 and check_single_answer(matches[0]):
                        answer = matches[0]
                        answer = json.loads(answer)
                        usage = body["usage"]
                        output = {
                            "custom_id": custom_id,
                            "model_used": model_used,
                            "answer": answer,
                            "usage": usage
                        }
                        json_outputs.append(output)
                    else:
                        print(f"invalid output custom id: {custom_id}")
                        invalid_outputs.append(json_output)
        if len(invalid_outputs) > 0 and process_invalid:
            invalid_ids = [c["custom_id"] for c in invalid_outputs]
            tasks = []
            with open(f"./data/batch_data/isq_batch.jsonl", 'r') as f:
                for l in f:
                    call = json.loads(l)
                    if call["custom_id"] in invalid_ids:
                        task = {"id": call["custom_id"], "prompt": call["body"]["messages"][0]["content"], "meta":{}}
                        tasks.append(task)
                retry_outputs = gpt_batch(tasks, model_name="gpt4")
                for i, r in enumerate(retry_outputs):
                    invalid_case = invalid_outputs[i]
                    if invalid_case["custom_id"] == r["id"]:
                        invalid_case["response"]["body"]["choices"][0]["message"]["content"] = r["output"]
                        body = invalid_case['response']["body"]
                        model_used = body["model"]
                        answer = r["output"]
                        usage = body["usage"]
                        usage["repaired"] = True
                        output = {
                            "custom_id": invalid_case['custom_id'],
                            "model_used": model_used,
                            "answer": answer,
                            "usage": usage
                        }
                        json_outputs.append(output)

    if len(json_outputs) > 0:
        with open(OUPUT_FOLDER + f'/{MODEL}_{MODE}_repaired_output.json', 'w') as f:
            json.dump(json_outputs, f, ensure_ascii=False)

    return json_outputs, invalid_outputs



def download_multiparts_batch_output(record_file):
    with open(record_file) as f:
        record = json.load(f)
        if len(record) > 0:
            full_outputs = []
            full_invalid_outputs = []
            for index, ids in record.items():
                part_outputs, invalid_outputs = download_batch_output(ids["batch_id"], int(index))
                full_outputs.extend(part_outputs)
                full_invalid_outputs.extend(invalid_outputs)
            if len(full_outputs) > 0:
                with open(OUPUT_FOLDER + f'/{MODEL}_{MODE}_full_output.json', 'w') as f:
                    json.dump(full_outputs, f, ensure_ascii=False)
            if len(full_invalid_outputs) > 0:
                with open(OUPUT_FOLDER + f'/{MODEL}_{MODE}_full_invalid_output.json', 'w') as f:
                    json.dump(full_invalid_outputs, f, ensure_ascii=False)

def process_invalid_cases(file):
    with open(file) as f:
        cases = json.load(f)
        batch_prompts = []
        invalid_cases_ids = [case["custom_id"] for case in cases]
        print(f"Invalid case number: {len(invalid_cases_ids)}")
        episodes = sorted(glob.glob(INPUT_FOLDER + "/*.xlsx"))
        for i, e in enumerate(episodes):
            if "~$" in e:
                continue
            eps_data = process_episode(e, MODEL)
            batch_prompts.extend(eps_data)
        batch_prompts = filter(lambda x: x["custom_id"] in invalid_cases_ids, batch_prompts)
        print(f"uploading invalid cases batch !")
        batch_data_file = OUPUT_FOLDER + f"/isq_batch_{MODE}_invalid_cases.jsonl"
        dicts_to_jsonl(batch_prompts, batch_data_file)
        batch_input_file_id, batch_id = post_openai_batch_request(batch_data_file, -1)
        print(f"batch id: {batch_id}")





if __name__ == "__main__":
    download_batch_output("batch_b8hu2jcktx1OJ85zXSvWc3CN", process_invalid=True)
    # check_status(LOG_FOLDER + "/batch_upload_record.json")
    # download_multiparts_batch_output(LOG_FOLDER + "/batch_upload_record.json")
    # process_invalid_cases(OUPUT_FOLDER + "/gpt-4o_train_full_invalid_output.json")