import pandas as pd
import numpy as np
import json
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from attributes_prompts import dialogue_acts
import krippendorff
from collections import Counter
import random

import warnings
warnings.filterwarnings("ignore")

CORPUS = "roundtable"
OUPUT_FOLDER = f"./data/{CORPUS}/output"

def main():
    with open(f"./data/{CORPUS}/agg/test.json", mode="r") as f:
        test_data = json.load(f)
    data = {
        "social motive": {
            "gpt":[],
            "human":[],
            "rl_gt":[],
            "rl_llm": [],
            "rl_vote":[]
        },
        "informational motive": {
            "gpt": [],
            "human": [],
            "rl_gt": [],
            "rl_llm": [],
            "rl_vote": []
        },
        "coordinative motive": {
            "gpt": [],
            "human": [],
            "rl_gt": [],
            "rl_llm": [],
            "rl_vote": []
        },
        "dialogue act": {
            "gpt": [],
            "human": [],
            "rl_gt": [],
            "rl_llm": [],
            "rl_vote": []
        },
        "target speaker":{
            "gpt": [],
            "human": [],
            "rl_gt": [],
            "rl_llm": [],
            "rl_vote": []
        }
    }

    eval_result = {}

    for num, i in enumerate(test_data):
        try:
            data["informational motive"]["gpt"].append( 1 if "informational motive" in i["answer"]["gpt"]["motives"] else 0)
            data["social motive"]["gpt"].append(1 if "social motive" in i["answer"]["gpt"]["motives"] else 0)
            data["coordinative motive"]["gpt"].append(1 if "coordinative motive" in i["answer"]["gpt"]["motives"] else 0)
            data["dialogue act"]["gpt"].append(i["answer"]["gpt"]["dialogue act"])
            data["target speaker"]["gpt"].append(i["answer"]["gpt"]['target speaker(s)'])

            im = int(i["answer"]["human"]["motives"]["informational motive"]["label"])
            data["informational motive"]["human"].append(im)
            im_votes = [v for k, v in i["answer"]["human"]["motives"]["informational motive"]["vote"].items()]
            data["informational motive"]["rl_vote"].extend(im_votes)
            data["informational motive"]["rl_gt"].extend([im] * len(im_votes))
            data["informational motive"]["rl_llm"].extend(
                [1 if "informational motive" in i["answer"]["gpt"]["motives"] else 0] * len(im_votes))


            cm = int(i["answer"]["human"]["motives"]["coordinative motive"]["label"])
            data["coordinative motive"]["human"].append(cm)
            cm_votes = [v for k, v in i["answer"]["human"]["motives"]["coordinative motive"]["vote"].items()]
            data["coordinative motive"]["rl_vote"].extend(cm_votes)
            data["coordinative motive"]["rl_gt"].extend([cm] * len(cm_votes))
            data["coordinative motive"]["rl_llm"].extend(
                [1 if "coordinative motive" in i["answer"]["gpt"]["motives"] else 0] * len(cm_votes))

            sm = int(i["answer"]["human"]["motives"]["social motive"]["label"])
            data["social motive"]["human"].append(sm)
            sm_votes = [v for k, v in i["answer"]["human"]["motives"]["social motive"]["vote"].items()]
            data["social motive"]["rl_vote"].extend(sm_votes)
            data["social motive"]["rl_gt"].extend([sm] * len(sm_votes))
            data["social motive"]["rl_llm"].extend(
                [1 if "social motive" in i["answer"]["gpt"]["motives"] else 0] * len(sm_votes))

            da = int(i["answer"]["human"]["dialogue act"]["label"])
            data["dialogue act"]["human"].append(da)
            da_votes = [v for k, v in i["answer"]["human"]["dialogue act"]["vote"].items()]
            data["dialogue act"]["rl_vote"].extend(da_votes)
            data["dialogue act"]["rl_gt"].extend([da] * len(da_votes))
            data["dialogue act"]["rl_llm"].extend([i["answer"]["gpt"]["dialogue act"]] * len(da_votes))

            ts = int(i["answer"]["human"]["target speaker(s)"]["label"])
            data["target speaker"]["human"].append(ts)
            ts_votes = [v for k, v in i["answer"]["human"]["target speaker(s)"]["vote"].items()]
            data["target speaker"]["rl_vote"].extend(ts_votes)
            data["target speaker"]["rl_gt"].extend([ts] * len(ts_votes))
            data["target speaker"]["rl_llm"].extend([i["answer"]["gpt"]['target speaker(s)']] * len(ts_votes))


        except Exception as e:
            pass

    for att, v in data.items():
        try:
            gpt = v["gpt"]
            human = v["human"]
            rl_gt = v["rl_gt"]
            rl_llm = v["rl_llm"]
            votes = v["rl_vote"]

            most_common_num = Counter(human).most_common(1)[0][0]
            maj_seq = [most_common_num] * len(human)

            eval_output = classification_report(human, gpt, output_dict=True)
            maj_report = classification_report(human, maj_seq, output_dict=True)
            rl_report = classification_report(rl_gt, votes, output_dict=True)
            gpt_rl_report = classification_report(votes, rl_llm,output_dict=True)

            random_microf1s = []
            for i in range(5):
                random_seq = random.choices(list(set(human)), k=len(human))
                macro_f1 = f1_score(human, random_seq, labels=list(set(human)), average='macro')
                random_microf1s.append(macro_f1)


            kp_alpha = krippendorff.alpha(reliability_data=[human, gpt], level_of_measurement="nominal")
            eval_result[att] = eval_output
            eval_result[att]["kp_alpha"] = kp_alpha
            eval_result[att]["random_macrof1"] = np.mean(random_microf1s)
        except Exception as e:
            pass

    with open(f"{CORPUS}_gpt_eval.json", mode="w") as f:
        json.dump(eval_result, f)

def get_gpt_human_comparison():
    with open("data/insq/agg/dev.json", mode="r") as f:
        test_data = json.load(f)
    data = []

    for num, i in enumerate(test_data):
        sample = {"id": i["id"], "topic": i["topic"], "speakers": ",\n".join(i["speakers"]),
                  "prior context": "".join([f"{s[0]} ({s[1]}): {s[2]} \n" for s in i["context"]["prior_context"]]),
                  "post context": "".join([f"{s[0]} ({s[1]}): {s[2]} \n" for s in i["context"]["post_context"]]),
                  "target": f"{i['target']['speaker']} ({i['target']['role']}): {i['target']['content']} \n"}
        sample["informational motive"] = int(i["answer"]["human"]["motives"]["informational motive"]["label"])
        sample["informational motive vote"] = i["answer"]["human"]["motives"]["informational motive"]["vote"]
        sample["informational motive gpt"] = 1 if "informational motive" in i["answer"]["gpt"]["motives"] else 0

        sample["social motive"] = int(i["answer"]["human"]["motives"]["social motive"]["label"])
        sample["social motive vote"] = i["answer"]["human"]["motives"]["social motive"]["vote"]
        sample["social motive gpt"] = 1 if "social motive" in i["answer"]["gpt"]["motives"] else 0

        sample["coordinative motive"] = int(i["answer"]["human"]["motives"]["coordinative motive"]["label"])
        sample["coordinative motive vote"] = i["answer"]["human"]["motives"]["coordinative motive"]["vote"]
        sample["coordinative motive gpt"] = 1 if "coordinative motive" in i["answer"]["gpt"]["motives"] else 0

        sample["dialogue act"] = int(i["answer"]["human"]["dialogue act"]["label"])
        sample["dialogue act vote"] = i["answer"]["human"]["dialogue act"]["vote"]
        sample["dialogue act gpt"] = i["answer"]["gpt"]["dialogue act"]

        sample["target speaker"] = int(i["answer"]["human"]["target speaker(s)"]["label"])
        sample["target speaker vote"] = i["answer"]["human"]["target speaker(s)"]["vote"]
        sample["target speaker gpt"] = i["answer"]["gpt"]['target speaker(s)']
        sample["gpt promp"] = i["answer"]["gpt"]["prompt"]
        sample["gpt reason"] = i["answer"]["gpt"]["reason"]
        data.append(sample)

    df = pd.DataFrame(data)
    df.to_excel("dev_valid_agg.xlsx")

def eval_single_task_gpt_prediction():
    eval_result = {}

    with open(f"./data/{CORPUS}/agg/test.json", mode="r") as f:
        test_data = json.load(f)
    data = {
        "social motive": {
            "gpt":[],
            "human":[]
        },
        "informational motive": {
            "gpt": [],
            "human": []
        },
        "coordinative motive": {
            "gpt": [],
            "human": []
        },
        "dialogue act": {
            "gpt": [],
            "human": []
        },
        "target speaker":{
            "gpt": [],
            "human": []
        }
    }
    single_tasks = ["informational motive", "social motive", "coordinative motive", "dialogue act", "target speaker"]
    gpt_single_tasks_result = {}
    for t in single_tasks:
        label_tag = "".join([w[0] for w in t.split(" ")])
        with open(OUPUT_FOLDER + f"/gpt-4o_test_{label_tag}_output.json", mode="r") as f:
            pred = json.load(f)
            gpt_single_tasks_result[t] = {p["custom_id"]: p for p in pred}

    for num, i in enumerate(test_data):
        id = i["id"]

        data["informational motive"]["human"].append(int(i["answer"]["human"]["motives"]["informational motive"]["label"]))
        data["social motive"]["human"].append(int(i["answer"]["human"]["motives"]["social motive"]["label"]))
        data["coordinative motive"]["human"].append(int(i["answer"]["human"]["motives"]["coordinative motive"]["label"]))
        data["dialogue act"]["human"].append(int(i["answer"]["human"]["dialogue act"]["label"]))
        data["target speaker"]["human"].append(int(i["answer"]["human"]["target speaker(s)"]["label"]))


        data["informational motive"]["gpt"].append(gpt_single_tasks_result["informational motive"][id]["answer"]["verdict"])
        data["social motive"]["gpt"].append(gpt_single_tasks_result["social motive"][id]["answer"]["verdict"])
        data["coordinative motive"]["gpt"].append(gpt_single_tasks_result["coordinative motive"][id]["answer"]["verdict"])
        data["dialogue act"]["gpt"].append(dialogue_acts[gpt_single_tasks_result["dialogue act"][id]["answer"]["dialogue act"]])
        data["target speaker"]["gpt"].append(int(gpt_single_tasks_result["target speaker"][id]["answer"]['target speaker(s)'].split(" ")[0]))

    for att, v in data.items():
        gpt = v["gpt"]
        human = v["human"]
        eval_output = classification_report(human, gpt, output_dict=True)
        kp_alpha = krippendorff.alpha(reliability_data=[human, gpt], level_of_measurement="nominal")
        eval_result[att] = eval_output
        eval_result[att]["kp_alpha"] = kp_alpha

    with open(f"gpt_{CORPUS}_single_task_eval.json", mode="w") as f:
        json.dump(eval_result, f)


if __name__ == "__main__":
    main()
    # get_gpt_human_comparison()
    # eval_single_task_gpt_prediction()