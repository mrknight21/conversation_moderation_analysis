import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import re
import json
from scipy import stats
import seaborn as sns
from collections import Counter
from attributes_prompts import dialogue_acts, dialogue_acts_decode, motive_decode
CORPUS = "roundtable"
MODE = "gpt"
DATA_FOLDER = f"./data/{CORPUS}/agg/"
RAW_TRAIN_DATA = f"./data/{CORPUS}/train_data/"
RAW_DEV_DATA = f"./data/{CORPUS}/dev_data/"
RAW_TEST_DATA = f"./data/{CORPUS}/test_data/"
FIG_FOLDER = "./figure/"

MOTIVES = ["i"]

def analyse_episode(df):
    stats = {
            "tokens":{
            "total":0,
            "mod": 0,
            "for": 0,
            "against": 0,
            "others": 0
            },
            "utterances_count":{
            "total":0,
            "mod": 0,
            "for": 0,
            "against": 0,
            "others": 0
            },
        "utterances_length":{
            "total": [],
            "mod": [],
            "for": [],
            "against": [],
            "others": []
        }
        }

    utterance = []

    for i, r in df.iterrows():
        if pd.notna(r.text):
            role = r.role
            if CORPUS == 'insq':
                roles = ["for", "against", "mod"]
            else:
                roles = ["speakers" , "mod"]
            if role not in roles:
                role = "others"
            tokens_count = len(str(r.text).split(" "))
            stats["tokens"]["total"] += tokens_count
            stats["tokens"][role] += tokens_count
            stats["utterances_length"]["total"].append(tokens_count)
            stats["utterances_length"][role].append(tokens_count)
            if int(r.id.split("_")[-1]) == 0 or "_" not in r.id:
                stats["utterances"]["total"] +=1
                stats["utterances"][role] +=1


    for k, v in stats["tokens"].items():
        if k == "total":
            continue
        total = stats["tokens"]["total"]
        stats["tokens"][k] = stats["tokens"][k] / total

    for k, v in stats["utterances_count"].items():
        if k == "total":
            continue
        total = stats["utterances"]["total"]
        stats["utterances"][k] = stats["utterances"][k] / total

    for k, v in stats["utterances_length"].items():
        if k == "total":
            continue
        total = stats["utterances"]["total"]
        stats["utterances"][k] = stats["utterances"][k] / total
    return stats

def generate_pairs_sequence(sequence):
    pairs = []
    for i, a in enumerate(sequence):
        for j, b in enumerate(sequence):
            if i == j:
                continue
            else:
                pairs.append([a, b])
    return pairs


def load_json_data(path):
    with open(path) as f:
        json_objs = json.load(f)
        return json_objs

def get_episode_meta(episode):
    meta_json_path = episode.replace(".xlsx", "_meta.json")
    meta = None
    with open(meta_json_path) as f:
        metas = json.load(f)
        if len(metas.values()) > 0:
            meta = list(metas.values())[0]
        meta["topic"] = " ".join(meta["topic"].split("_")[2:])
    return meta

def generate_intervals(n, m):
    interval_size = n // m
    intervals = []
    for i in range(m):
        start = i * interval_size
        end = (i + 1) * interval_size - 1
        if i == m - 1:  # Adjust the last interval to include the remainder
            end = n - 1
        intervals.append((start, end))
    return intervals


def find_intercepted_interval_indexes(range_tuple, intervals):
    i, j = range_tuple
    intercepted_indexes = []

    for index, interval in enumerate(intervals):
        start, end = interval
        # Check if intervals intersect
        if start <= j and end >= i:
            intercepted_indexes.append(index)

    return intercepted_indexes

def get_test_test(a_mean, a_std, b_mean, b_std):
    columns = a_mean.columns
    indexs = a_mean.index
    t_test_result = pd.DataFrame(False, index=indexs, columns=columns)

    for c in columns:
        for i in indexs:

            c_total = c == "total"
            i_total = i == "total"

            if c_total and i_total:
                continue
            elif c_total or i_total:
                a_num = a_mean.at["total", "total"]
                b_num = b_mean.at["total", "total"]
            else:
                a_num = int(a_mean.at[i, "total"] * a_mean.at["total", "total"])
                b_num = int(b_mean.at[i, "total"] * b_mean.at["total", "total"])

            # Given data
            mean_A = a_mean.at[i, c]
            std_A = a_std.at[i, c]

            mean_B = b_mean.at[i, c]
            std_B = b_std.at[i, c]

            t_statistic, p_value = stats.ttest_ind_from_stats(
                mean1=mean_A, std1=std_A, nobs1=a_num,
                mean2=mean_B, std2=std_B, nobs2=b_num,
                equal_var=False  # Welch's t-test
            )

            significance = p_value < 0.05
            t_test_result.at[i, c] = significance

    return t_test_result


def get_motive_dialogue_act_matrix_episode_breakdown(label_sequence, index=None, col=None):
    eps_sqs_dict = {}
    eps_dfs = []
    for seq in label_sequence:
        seq = [s for s in seq if s["role"] is not None]
        eps = seq[0]["episode"]
        if eps not in eps_sqs_dict:
            eps_sqs_dict[eps] = []
        eps_sqs_dict[eps].append(seq)

    for eps, seqs in eps_sqs_dict.items():
        if eps == '_Shopping_Boycott_agg':
            pass
        co_occurrence_counts, labels_count = get_motive_dialogue_act_matrix(seqs, index, col, add_total=True)
        if co_occurrence_counts.isnull().values.any():
            print("DataFrame contains NaN values.")
        eps_dfs.append(co_occurrence_counts)

    # Stack the DataFrames into a 3D array (n, rows, cols)
    array_3d = np.array([df.values for df in eps_dfs])

    # Calculate the means and stds for each cell
    mean_array = np.mean(array_3d, axis=0)
    std_array = np.std(array_3d, axis=0)

    # Create DataFrames from the results
    mean_df = pd.DataFrame(mean_array, columns=eps_dfs[0].columns, index=eps_dfs[0].index)
    std_df = pd.DataFrame(std_array, columns=eps_dfs[0].columns, index=eps_dfs[0].index)

    return mean_df, std_df


def get_motive_dialogue_act_matrix(label_sequence, index=None, col=None, add_total=True):
    items = []
    labels_count = {"da": [0, 0, 0, 0, 0, 0], "m": [0, 0, 0], "total": 0}
    for sents in label_sequence:
        for sent in sents:
            if "transition" in sent["labels"]:
                continue
            labels_count["total"] += 1
            dialogue_act, motives = sent["labels"]["da"], sent["labels"]["m"]
            try:
                labels_count["da"][int(dialogue_act)] += 1
            except Exception as e:
                dialogue_act = str(int(float(dialogue_act)))
                labels_count["da"][int(dialogue_act)] += 1
            if motives:
                for m in motives:
                    m_index = "ics".index(m)
                    labels_count["m"][m_index] += 1
                    items.append((dialogue_acts_decode[dialogue_act], motive_decode[m]))

    if not index:
        first_dim = [item[0] for item in items]
        unique_first_dim = sorted(set(first_dim))
        index = unique_first_dim

    if not col:
        second_dim = [item[1] for item in items]
        unique_second_dim = sorted(set(second_dim))
        col = unique_second_dim

    # Step 2: Count Co-occurrences
    co_occurrence_counts = pd.DataFrame(0, index=index, columns=col)

    for item in items:
        co_occurrence_counts.at[item[0], item[1]] += 1

    co_occurrence_counts = co_occurrence_counts[[motive_decode['i'], motive_decode['c'], motive_decode['s']]]
    co_occurrence_counts.columns = ["IM", "CM", "SM"]
    co_occurrence_counts = co_occurrence_counts.T
    co_occurrence_counts = co_occurrence_counts.div(labels_count['m'], axis=0)

    co_occurrence_counts = co_occurrence_counts.round(4)
    co_occurrence_counts = co_occurrence_counts.fillna(0)
    # co_occurrence_counts["total"] = [c / labels_count['total'] for c in labels_count['da']]
    if add_total:
        co_occurrence_counts["total"] = [c /  labels_count['total'] for c in labels_count['m'] ]
        dialogue_act_count = [c / labels_count['total'] for c in labels_count['da']]
        last_row = pd.DataFrame([dialogue_act_count+ [labels_count['total']]], columns=co_occurrence_counts.columns)
        last_row.index = ["total"]
        co_occurrence_counts = pd.concat([co_occurrence_counts, last_row])

    co_occurrence_counts.columns = [re.sub('[^A-Za-z]+', '', c)[:4] if c != "total" else "total" for c in
                                    co_occurrence_counts.columns]

    return co_occurrence_counts, labels_count

def generate_utterance_label(label_sequence, meta, prior_mod_info=None, post_mod_info=None):
    speakers_option = [s.split(" (") for s in meta["speakers"]]
    speakers_option = {s[0]: s[1].replace(")", "") for s in speakers_option}
    text = " ".join([u["text"] for u in label_sequence])

    unique_labels = set([str(l["labels"]["da"]) for l in label_sequence])
    motives = set([c for l in label_sequence for c in l["labels"]["m"]])
    unique_pos = set([j for i in label_sequence for j in i["interval_index"]])
    utterance_label = "-".join(sorted(unique_labels))
    motives_label = "".join(sorted(motives))
    utterance_pos = sorted(unique_pos)
    target_speakers = sorted(set([str(l["labels"]["ts"]) for l in label_sequence]))
    try:
        target_speakers = [speakers_option[s] for s in target_speakers]
        individual_target_speakers = [s for s in target_speakers if s not in ["Unknown", "Self", "Everyone", "Audience", "Against team", "Support team", "All speakers"]]

    except Exception as e:
        pass

    utt_length = sum([l["count"] for l in label_sequence])



    prior_speaker = None
    post_speaker = None
    if prior_mod_info:
        prior_speaker = prior_mod_info["speaker"].split(" (")[0]
        if prior_speaker:
            prior_speaker = [s for i, s in speakers_option.items() if prior_speaker in s]
            if prior_speaker:
                prior_speaker = prior_speaker[0]
    if post_mod_info:
        post_speaker = post_mod_info["speaker"].split(" (")[0]
        if post_speaker:
            post_speaker = [s for i, s in speakers_option.items() if post_speaker in s]
            if post_speaker:
                post_speaker = post_speaker[0]


    for i, s in enumerate(label_sequence):

        target_speaker = meta["speakers"][int(s["labels"]["ts"])]

        responding = False
        responded = False

        if "team" in target_speaker:
            team = "for" if "Support" in target_speaker else "against"
            if prior_speaker:
                responding = team in prior_speaker
            if post_speaker:
                responded = team in post_speaker
        else:
            if prior_speaker:
                responding = prior_speaker == target_speaker
            if post_speaker:
                responded = post_speaker == target_speaker

        specific = not any([w in target_speaker.lower() for w in ["everyone", "self", "unknown", "audience", "team", "speakers", "all"]])

        label_sequence[i]["responding"] = responding
        label_sequence[i]["responded"] = responded
        label_sequence[i]["specific"] = specific

    if individual_target_speakers:
        specific = True
    else:
        specific = False

    if prior_mod_info:
        proactive = not any([s == prior_speaker for s in target_speakers])
    else:
        proactive = True

    if post_mod_info and individual_target_speakers:
        interactive = any([ s == post_speaker for s in individual_target_speakers])
    else:
        interactive = False

    if post_mod_info and proactive :
        position_rotate = (prior_mod_info["role"] != post_mod_info["role"]) and (prior_mod_info["role"] != "audience" and post_mod_info["role"] != "audience")
    else:
        position_rotate = False

    return {"split": label_sequence[0]["split"], "episode": label_sequence[0]["episode"], "speaker": label_sequence[0]["speaker"], "text": text, "count": utt_length, "role": label_sequence[0]["role"], "interval_index": utterance_pos,
            "sentence_count": len(label_sequence), "individual target speaker count": len(individual_target_speakers),
            "labels":{"da": utterance_label, "ts": target_speakers, "m": motives_label,
                      "proactive": proactive, "interactive": interactive, "specific": specific, "position_rotate": position_rotate}}, label_sequence


def get_transition_matrix(sequences, filter_funcs=[], pair_filter_funcs=[], name=None):
    transitions = {}
    unique_states = set()
    for states in sequences:
        if filter_funcs:
            states = [s for s in states if all([f(s) for f in filter_funcs])]
        for (s1, s2) in zip(states[:-1], states[1:]):
            if s1["role"] == "host" or s2["role"] == "host":
                continue
            if pair_filter_funcs:
                if not all([f((s1, s2)) for f in pair_filter_funcs]):
                    continue
            if "da" in s1["labels"]:
                s1 = s1["labels"]["da"].split("-")
            else:
                s1 = [s1["labels"]["transition"]]

            if "da" in s2["labels"]:
                s2 = s2["labels"]["da"].split("-")
            else:
                s2 = [s2["labels"]["transition"]]

            for i in s1:
                if i in dialogue_acts_decode:
                    i = dialogue_acts_decode[i]
                if i not in transitions:
                    unique_states.add(i)
                    transitions[i] = {}
                for j in s2:
                    if j in dialogue_acts_decode:
                        j = dialogue_acts_decode[j]
                    if j not in transitions[i]:
                        unique_states.add(j)
                        transitions[i][j] = 0
                    transitions[i][j] += 1

    unique_states = sorted(unique_states)

    # Initialize the transition matrix with zeros
    transition_matrix = pd.DataFrame(0, index=unique_states, columns=unique_states)

    # Fill the transition matrix with transition counts
    for s1 in transitions:
        for s2 in transitions[s1]:
            transition_matrix.at[s1, s2] = transitions[s1][s2]

    # Step 2: Calculate Probabilities
    transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)

    # Step 3: Plot the Transition Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(transition_matrix, annot=True, cmap="Blues", fmt=".2f", cbar=True)
    plt.title("State Transition Matrix")
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.savefig(FIG_FOLDER + name, format='pdf', dpi=300, bbox_inches="tight")
    plt.show()



def get_label_sequences(corpus, with_text=False, splits=["test", "dev", "train"], mode="gpt"):

    agg_dict = {}
    data_source = glob.glob(f"./data/{corpus}/agg/" + "/*.json")
    for d in data_source:
        data = {l["id"]: l for l in load_json_data(d)}
        key = d.split("/")[-1].replace(".json", "")
        agg_dict[key] = data

    stats = {"overall": {}}
    episodes = {}
    for k in agg_dict.keys():
        if k not in splits:
            continue
        if k == "train" and MODE != "human":
            episodes[k] = glob.glob(f"./data/{corpus}/train_data/" + "/*.xlsx")
            stats[k] = {}
        elif k == "dev":
            episodes[k] = glob.glob(f"./data/{corpus}/dev_data/" + "/*.xlsx")
            stats[k] = {}
        elif k == "test":
            episodes[k] = glob.glob(f"./data/{corpus}/test_data/" + "/*.xlsx")
            stats[k] = {}

    inter_utterance_sequence = []
    intra_utterance_sequence = []

    utterance_stats = []
    eps_speaker_stats = []
    mod_sents = []

    vote_pairs = {
        "da": [],
        "ts": []
    }

    for split, v in episodes.items():
        for episode in v:
            episode_inter_utterance_sequence = []
            debate = pd.read_excel(episode, index_col=0)
            meta = get_episode_meta(episode)
            total_tokens_count = np.sum([len(str(s).split(" ")) for s in debate["text"].dropna().tolist()])
            token_count_intervals = generate_intervals(total_tokens_count, 40)
            utterance_cache = []
            prior_mod_cache = {
                "speaker": "",
                "role": "",
                "index": -1
            }
            accumulate_tokens_count = 0
            eps_speaker_stats.append({"split": split, "episode": episode.split("/")[-1].replace(".xlsx", ""), "speaker_count": len(meta["speakers"]) - 7, "token_count": total_tokens_count})

            for i, r in debate.iterrows():
                if pd.isna(r.text):
                    continue
                text = str(r.text)
                tokens_count = len(text.split(" "))
                interval_index = find_intercepted_interval_indexes((accumulate_tokens_count, accumulate_tokens_count+tokens_count), token_count_intervals)
                accumulate_tokens_count += tokens_count
                if r.role == "mod":
                    id = episode.split("/")[-1].replace(".xlsx", "") + "_" + r.id
                    if split == "train":
                        answer = agg_dict[split][id]["answer"]
                        label = str(dialogue_acts[answer["dialogue act"]])
                        target_speaker = answer['target speaker(s)']
                        target_speaker = int(target_speaker.split(" ")[0])
                        motive_code = "".join([m[0] for m in answer["motives"]])
                    else:
                        answer = agg_dict[split][id]["answer"]["gpt"]
                        label = str(answer["dialogue act"])
                        target_speaker = answer['target speaker(s)']
                        target_speaker = int(target_speaker)

                        ##get votes
                        human_da = agg_dict[split][id]["answer"]["human"]["dialogue act"]["label"]
                        human_ts = agg_dict[split][id]["answer"]["human"]["target speaker(s)"]["label"]
                        human_im = agg_dict[split][id]["answer"]["human"]["motives"]["informational motive"]["label"]
                        human_cm = agg_dict[split][id]["answer"]["human"]["motives"]["coordinative motive"]["label"]
                        human_sm = agg_dict[split][id]["answer"]["human"]["motives"]["social motive"]["label"]

                        da_votes = list(agg_dict[split][id]["answer"]["human"]["dialogue act"]["vote"].values())
                        ts_votes = list(agg_dict[split][id]["answer"]["human"]["target speaker(s)"]["vote"].values())
                        im_votes = list(agg_dict[split][id]["answer"]["human"]["motives"]["informational motive"]["vote"].values())
                        cm_votes = list(agg_dict[split][id]["answer"]["human"]["motives"]["coordinative motive"]["vote"].values())
                        sm_votes = list(agg_dict[split][id]["answer"]["human"]["motives"]["social motive"]["vote"].values())

                        da_votes_pairs = generate_pairs_sequence(da_votes)
                        ts_votes_pairs = generate_pairs_sequence(ts_votes)

                        vote_pairs["da"].extend(da_votes_pairs)
                        vote_pairs["ts"].extend(ts_votes_pairs)

                        if mode == "human":
                            answer = agg_dict[split][id]["answer"]["human"]
                            label = str(answer["dialogue act"]["label"])
                            target_speaker = answer["target speaker(s)"]["label"]
                            target_speaker = int(target_speaker)
                            motive_code = "".join([m[0] for m, v in answer["motives"].items() if v["label"] == 1])
                        else:
                            answer = agg_dict[split][id]["answer"]["gpt"]
                            label = str(answer["dialogue act"])
                            target_speaker = answer['target speaker(s)']
                            target_speaker = int(target_speaker)
                            motive_code = "".join([m[0] for m in answer["motives"]])

                    mod_sents.append({"text": text,
                                      "prob": label == "0",
                                      "conf": label == "1",
                                      "inst": label == "2",
                                      "inte": label == "3",
                                      "supp": label == "4",
                                      "util": label == "5",
                                      "im": "i" in motive_code,
                                      "cm": "c" in motive_code,
                                      "sm": "s" in motive_code })
                    utterance_cache.append({"split": split, "episode": episode.split("/")[-1].replace(".xlsx", ""), "text": text, "speaker": r.speaker, "role": r.role, "interval_index": interval_index, "count": tokens_count, "labels":{"da":str(label), "m": motive_code, "ts": target_speaker}})
                else:
                    speaker = r.speaker
                    label = "transition"
                    if r.role == prior_mod_cache["role"]:
                        if speaker == prior_mod_cache["speaker"]:
                            if i - prior_mod_cache["index"] == 1:
                                interval_index.extend(episode_inter_utterance_sequence[-1]["interval_index"])
                                interval_index = set(interval_index)
                                episode_inter_utterance_sequence[-1]["interval_index"] = sorted(interval_index)
                                episode_inter_utterance_sequence[-1]["sentence_count"] += len(text.split(". "))
                                episode_inter_utterance_sequence[-1]["count"] += tokens_count
                                prior_mod_cache['index'] = i
                                continue
                            label = "speaker cont."
                        elif corpus == "insq":
                            label = "stance cont."

                    cur_utt_info = {"split": split, "episode": episode.split("/")[-1].replace(".xlsx", ""), "speaker": r.speaker, "text": text, "sentence_count": len(text.split(". ")),
                                    "role": r.role, "interval_index": interval_index, "count": tokens_count, "labels":{"transition": label}}

                    if utterance_cache:
                        agg_mod_utt, utterance_cache = generate_utterance_label(utterance_cache, meta, prior_mod_cache, cur_utt_info)
                        episode_inter_utterance_sequence.append(agg_mod_utt)
                        utterance_stats.append(agg_mod_utt)
                        intra_utterance_sequence.append(utterance_cache)
                        utterance_cache = []

                    prior_mod_cache["speaker"] = speaker
                    prior_mod_cache["role"] = r.role
                    prior_mod_cache["index"] = i
                    episode_inter_utterance_sequence.append(cur_utt_info)
                    utterance_stats.append(cur_utt_info)

            if utterance_cache:
                agg_mod_utt, utterance_cache = generate_utterance_label(utterance_cache, meta, prior_mod_cache, cur_utt_info)
                episode_inter_utterance_sequence.append(agg_mod_utt)
                utterance_stats.append(agg_mod_utt)
                intra_utterance_sequence.append(utterance_cache)
                utterance_cache = []
            inter_utterance_sequence.append(episode_inter_utterance_sequence)

    for i in range(len(inter_utterance_sequence)):
        inter_utterance_sequence[i] = [{"speaker": None, "role": None, "interval_index": [0], 'individual target speaker count':0, "labels":{"transition": "#START"}}] + inter_utterance_sequence[
            i] + [{"speaker": None, "role": None, "interval_index": [40-1], 'individual target speaker count':0,  "labels":{"transition": "#END"}}]

    for i in range(len(intra_utterance_sequence)):
        intra_utterance_sequence[i] = [{"speaker": None, "role": None, "interval_index": [0], "labels":{"transition": "#START"}}] + intra_utterance_sequence[
            i] + [{"speaker": None, "role": None, "interval_index": [intra_utterance_sequence[i][-1]["interval_index"][-1]], "labels":{"transition": "#END"}}]

    # eps_df = pd.DataFrame(eps_speaker_stats)
    # eps_df.to_csv(f"{corpus}_eps_stats.csv")
    #
    # utt_df = pd.DataFrame(utterance_stats).drop(["speaker", "interval_index", "labels"], axis=1)
    # utt_df.to_csv(f"{corpus}_utt_stats.csv")

    return inter_utterance_sequence, intra_utterance_sequence


mod_filter = lambda x: x["role"] == "mod"

def motive_filter_factor(motives):
    def motive_filer(pair):
        for p in pair:
            if "m" in p["labels"]:
                if any([m in motives for m in p["labels"]["m"]]):
                    return True
        return False
    return motive_filer

def position_filter_factor(positions_range):
    def position_filter(state):
        position_index = state["interval_index"]
        start, end = positions_range
        # Check if intervals intersect
        for i in position_index:
            if start <= i <= end:
                return True
        return False
    return position_filter



def main():
    index = ['0.probing', '1.confronting', '2.instruction', '3.interpretation', '4.supplement', '5.utility']
    col = ['informational', 'coordinative', 'social']

    insq_gpt_inter, insq_gpt_intra = get_label_sequences("insq", splits=["dev", "test"])
    rt_gpt_inter, rt_gpt_intra = get_label_sequences("roundtable", splits=["dev", "test"])

    insq_human_inter, insq_human_intra = get_label_sequences("insq", splits=["dev", "test"], mode="human")
    rt_human_inter, rt_human_intra = get_label_sequences("roundtable", splits=["dev", "test"], mode="human")

    # co_occurrence_counts, labels_count = get_motive_dialogue_act_matrix(intra_utterance_sequence)

    # insq
    insq_human_mean, insq_human_std = get_motive_dialogue_act_matrix_episode_breakdown(insq_human_intra, index=index, col=col)
    insq_gpt_mean, insq_gpt_std = get_motive_dialogue_act_matrix_episode_breakdown(insq_gpt_intra, index=index, col=col)

    insq_result = get_test_test(insq_human_mean, insq_human_std, insq_gpt_mean, insq_gpt_std)

    # rt
    rt_human_mean, rt_human_std = get_motive_dialogue_act_matrix_episode_breakdown(rt_human_intra, index=index, col=col)
    rt_gpt_mean, rt_gpt_std = get_motive_dialogue_act_matrix_episode_breakdown(rt_gpt_intra, index=index, col=col)

    rt_result = get_test_test(rt_human_mean, rt_human_std, rt_gpt_mean, rt_gpt_std)
    pass






if __name__ == "__main__":
    main()