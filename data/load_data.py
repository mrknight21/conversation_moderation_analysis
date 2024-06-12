import pandas as pd
import glob
import json
from run_llm_apis import get_episode_meta, check_single_answer, gpt_single, gpt4
from attributes_prompts import construct_prompt_unit, annotators, dialogue_acts

MODE = "train"
MODEL = "gpt-4o"
INPUT_FOLDER = f"./{MODE}_data"
OUPUT_FOLDER = f"./batch_data/{MODE}"
PRIOR_CONTEXT_SIZE = 5
POST_CONTEXT_SIZE = 2


def process_episode(episode):
    meta = get_episode_meta(episode)
    debate = pd.read_excel(episode, index_col=0)

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
    return task_instances

def load_train_data(files):
    data = {}
    for file in files:
        with open(file) as f:
            instances = json.load(f)
            for i in instances:
                data[i["custom_id"]] = {
                    "id": i["custom_id"],
                    "answer": i["answer"],
                    "model_used": i["model_used"],
                    "usage": i["usage"]
                }
                # data[i["custom_id"]]["answer"]["dialogue act"] = dialogue_acts[data[i["custom_id"]]["answer"]["dialogue act"]]
                # data[i["custom_id"]]["answer"]["target speaker(s)"] = int(data[i["custom_id"]]["answer"]["target speaker(s)"][0])

    episodes = glob.glob(INPUT_FOLDER + "/*.xlsx")
    for episode in episodes:
        instances = process_episode(episode)
        for i in instances:
            id = episode.split("/")[-1].replace(".xlsx", "") + "_" + i['id']
            if id in data:
                data[id]["topic_id"] = "_".join(id.split("_")[:-2])
                data[id]["instance_id"] = i["id"]
                data[id]["topic"] = i["meta"]["topic"]
                data[id]["speakers"] = i["meta"]["speakers"]
                data[id]["context"] = i["context"]
                data[id]["target"] = {"speaker": i["target"][0], "role": i["target"][1], "content": i["target"][2]}
                data[id]["prompt"] = i["prompt"]
                gpt_answer = json.dumps(data[id]["answer"])
                while not check_single_answer(gpt_answer, i):
                    print(f"repair id:{id}")
                    gpt_answer = gpt_single(i["prompt"], gpt4)
                data[id]["answer"] = json.loads(gpt_answer)
            else:
                print(f"missing case: {id}")
    train_set = list(data.values())
    # Convert and write JSON object to file
    with open(OUPUT_FOLDER + f"/train.json", "w") as outfile:
        json.dump(train_set, outfile)

def hide_voter_identity(votes):
    votes = json.loads(votes.replace("'", '"'))
    new_votes = {"annotator_"+str(annotators[k]): v for k, v in votes.items()}
    return new_votes
def aggregate_human_model_outputs(gpt_output_file):
    data = {}
    with open(gpt_output_file) as f:
        instances = json.load(f)
        for i in instances:
            data[i["custom_id"]] = {
                "id": i["custom_id"],
                "answer": i["answer"],
                "model_used": i["model_used"],
                "usage": i["usage"]
            }

    episodes = glob.glob(INPUT_FOLDER + "/*.xlsx")
    unprocess_row = []
    for episode in episodes:
        ep_name = episode.split("/")[-1].replace(".xlsx", "")
        meta = get_episode_meta(episode)
        debate = pd.read_excel(episode, index_col=0)
        for i, r in debate.iterrows():
            if r.role == "mod":
                id = ep_name + "_" + r.id
                if id in data:
                    utt_id = int(r.id.split("_")[0])
                    sent_id = int(r.id.split("_")[1])
                    prior_context_mask = debate.id.apply(lambda x: not (
                            not (utt_id - PRIOR_CONTEXT_SIZE <= int(x.split("_")[0]) < utt_id) and not (
                                int(x.split("_")[0]) == utt_id and \
                                int(x.split("_")[1]) < sent_id)))
                    post_context_mask = debate.id.apply(lambda x: not (
                            not (utt_id + POST_CONTEXT_SIZE >= int(x.split("_")[0]) > utt_id) and not (
                                int(x.split("_")[0]) == utt_id and \
                                int(x.split("_")[1]) > sent_id)))
                    context = {
                        "prior_context": [],
                        "post_context": []
                    }

                    if len(debate[prior_context_mask]) > 0:
                        context["prior_context"] = [(v.speaker, v.role, v.text) for i, v in
                                                    debate[prior_context_mask].iterrows()]

                    if len(debate[post_context_mask]) > 0:
                        context["post_context"] = [(v.speaker, v.role, v.text) for i, v in
                                                   debate[post_context_mask].iterrows()]
                    instance = {
                        "id": r.id,
                        "meta": meta,
                        "context": context,
                        "target": (r.speaker, r.role, r.text)
                    }
                    prompt = construct_prompt_unit(instance)

                    data[id]["topic_id"] = "_".join(id.split("_")[:-2])
                    data[id]["instance_id"] = r.id
                    data[id]["topic"] = meta["topic"]
                    data[id]["speakers"] = meta["speakers"]
                    data[id]["context"] = context
                    data[id]["target"] = {"speaker": r.speaker, "role": r.role, "content": r.text}

                    # human answ
                    human_answer = {
                        "motives":{
                            "informational motive": {
                                "label": r["informational motive"],
                                "vote": hide_voter_identity(r["informational motive vote"])
                            },
                            "social motive": {
                                "label": r["social motive"],
                                "vote": hide_voter_identity(r["social motive vote"])
                            },
                            "coordinative motive": {
                                "label": r["coordinative motive"],
                                "vote": hide_voter_identity(r["coordinative motive vote"])
                            },
                        },
                        "dialogue act": {
                            "label": int(r["dialogue act"]),
                            "vote": hide_voter_identity(r["dialogue act vote"])
                        },
                        "target speaker(s)":{
                            "label": int(r["target speaker"]),
                            "vote": hide_voter_identity(r["target speaker vote"])
                        }
                    }
                    try:
                        gpt_answer = json.dumps(data[id]["answer"])
                        while not check_single_answer(gpt_answer, instance):
                            gpt_answer = gpt_single(prompt, gpt4)
                        gpt_answer = json.loads(gpt_answer)
                        data[id]["answer"] = {"gpt": gpt_answer, "human": human_answer}
                        data[id]["answer"]["gpt"]["dialogue act"] = dialogue_acts[data[id]["answer"]["gpt"]["dialogue act"]]
                        gpt_pred_speaker = data[id]["answer"]["gpt"]["target speaker(s)"]
                        data[id]["answer"]["gpt"]["target speaker(s)"] = [i for i, s in enumerate(meta["speakers"]) if gpt_pred_speaker.lower() in s.lower()][0]
                        data[id]["answer"]["gpt"]["prompt"] = prompt
                    except Exception as e:
                        unprocess_row.append(id)
                else:
                    unprocess_row.append(id)
    if len(unprocess_row) > 0:
        print(unprocess_row)
    data = list(data.values())
    # Convert and write JSON object to file
    with open(OUPUT_FOLDER + f"/{MODE}.json", "w") as outfile:
        json.dump(data, outfile)

def convert_gpt_human_data(gpt_annotated_folder):

    episodes = glob.glob(INPUT_FOLDER + "/*.xlsx")
    data = {}

    for episode in episodes:
        ep_name = episode.split("/")[-1].replace(".xlsx", "")
        meta = get_episode_meta(episode)
        debate =pd.read_excel(gpt_annotated_folder + "/" + ep_name + "_dev_llm.xlsx", index_col=0)
        unprocessed_row = []
        for i, r in debate.iterrows():
            if r.role == "mod":
                id = ep_name + "_" + r.id
                utt_id = int(r.id.split("_")[0])
                sent_id = int(r.id.split("_")[1])
                prior_context_mask = debate.id.apply(lambda x: not (
                        not (utt_id - PRIOR_CONTEXT_SIZE <= int(x.split("_")[0]) < utt_id) and not (
                        int(x.split("_")[0]) == utt_id and \
                        int(x.split("_")[1]) < sent_id)))
                post_context_mask = debate.id.apply(lambda x: not (
                        not (utt_id + POST_CONTEXT_SIZE >= int(x.split("_")[0]) > utt_id) and not (
                        int(x.split("_")[0]) == utt_id and \
                        int(x.split("_")[1]) > sent_id)))
                context = {
                    "prior_context": [],
                    "post_context": []
                }

                if len(debate[prior_context_mask]) > 0:
                    context["prior_context"] = [(v.speaker, v.role, v.text) for i, v in
                                                debate[prior_context_mask].iterrows()]

                if len(debate[post_context_mask]) > 0:
                    context["post_context"] = [(v.speaker, v.role, v.text) for i, v in
                                               debate[post_context_mask].iterrows()]
                instance = {
                    "id": r.id,
                    "meta": meta,
                    "context": context,
                    "target": (r.speaker, r.role, r.text)
                }
                prompt = construct_prompt_unit(instance)

                data[id] = {}
                data[id]["id"] = id
                data[id]["topic_id"] = "_".join(id.split("_")[:-2])
                data[id]["instance_id"] = r.id
                data[id]["topic"] = meta["topic"]
                data[id]["speakers"] = meta["speakers"]
                data[id]["context"] = context
                data[id]["target"] = {"speaker": r.speaker, "role": r.role, "content": r.text}

                human_answer = {
                    "motives": {
                        "informational motive": {
                            "label": r["informational motive"],
                            "vote": hide_voter_identity(r["informational motive vote"])
                        },
                        "social motive": {
                            "label": r["social motive"],
                            "vote": hide_voter_identity(r["social motive vote"])
                        },
                        "coordinative motive": {
                            "label": r["coordinative motive"],
                            "vote": hide_voter_identity(r["coordinative motive vote"])
                        },
                    },
                    "dialogue act": {
                        "label": r["dialogue act"],
                        "vote": hide_voter_identity(r["dialogue act vote"])
                    },
                    "target speaker(s)": {
                        "label": r["target speaker"],
                        "vote": hide_voter_identity(r["target speaker vote"])
                    }
                }

                try:
                    gpt_answer = {"motives":[m for m in ["informational motive", "social motive", "coordinative motive"] if r[m + " llm"] == 1],
                                  "dialogue act": r["dialogue act llm"],
                                  "target speaker(s)": r["target speaker llm"],
                                  "reason": r["llm reason"]}
                    gpt_answer = json.dumps(gpt_answer)
                    while not check_single_answer(gpt_answer, instance):
                        gpt_answer = gpt_single(prompt, gpt4)
                    gpt_answer = json.loads(gpt_answer)
                    data[id]["answer"] = {"gpt": gpt_answer, "human": human_answer}
                    data[id]["answer"]["gpt"]["dialogue act"] = dialogue_acts[data[id]["answer"]["gpt"]["dialogue act"]]
                    gpt_pred_speaker = data[id]["answer"]["gpt"]["target speaker(s)"]
                    data[id]["answer"]["gpt"]["target speaker(s)"] = \
                    [i for i, s in enumerate(meta["speakers"]) if gpt_pred_speaker.lower() in s.lower()][0]
                    data[id]["answer"]["gpt"]["prompt"] = prompt
                except Exception as e:
                    unprocessed_row.append(id)
    if len(unprocessed_row) > 0:
        print(len(unprocessed_row))
    data = list(data.values())
    # Convert and write JSON object to file
    with open(OUPUT_FOLDER + f"/{MODE}.json", "w") as outfile:
        json.dump(data, outfile)







if __name__ == "__main__":
    files = [OUPUT_FOLDER + "/gpt-4o_train_full_output.json", OUPUT_FOLDER + "/gpt-4o_train_repaired_output.json"]
    load_train_data(files)
    # aggregate_human_model_outputs(OUPUT_FOLDER + "/gpt-4o_test.json")
    # convert_gpt_human_data("./gpt_annotated_data")
