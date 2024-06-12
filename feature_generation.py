import argparse
import json
import random
import pandas as pd

from nltk.tokenize import sent_tokenize
from convokit import Corpus, download

role_ratio = {
    "mod": 0.5,
    "for": 0.25,
    "against": 0.25
}

role_dict = {
        "mod": 0,
        "for": 1,
        "against": 2,
    }

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_utt_num",
        default=10,
        type=int
    )

    parser.add_argument(
        "--max_input_length",
        default=2048,
        type=int
    )

    parser.add_argument(
        "--context_utt_num",
        default=10,
        type=int
    )

    parser.add_argument(
        "--max_dialogue_num",
        default=-1,
        type=int
    )

    parser.add_argument(
        "--data_dir",
        default="./data",
        type=str
    )
    return parser.parse_args()


def test_train_split(corpus, test_size):
    convs = [(k, v) for k, v in corpus.conversations.items()]
    train_size = len(convs) - test_size
    if train_size <= 0:
        return [], []
    test_debates = random.sample(convs, test_size)
    test_debates_ids = [d[0] for d in test_debates]
    train_debates = [d for d in convs if d[0] not in test_debates_ids]
    return test_debates, train_debates

def generate_dialogue_features(args, conv):
    pass

def validate_instance(answer, dialogue_history, context_length):
    flag = False
    ans_speaker = answer["speaker"]
    ans_utt_id = answer["id"].split("_")[0]
    recent_context = []
    utt_ids = []
    speakers = set()
    dialogue_history.reverse()
    role_counts = [0, 0, 0]
    other_roles = False
    for sent in dialogue_history[1:]:
        role_id = role_dict.get(sent["role"], -1)
        if role_id >= 0:
            role_counts[role_id] += 1
        else:
            other_roles = True
        sent_speaker = sent["speaker"]
        speakers.add(sent_speaker)
        utt_id = sent["id"].split("_")[0]
        if utt_id not in utt_ids and utt_id != ans_utt_id:
            if len(utt_ids) > context_length:
                break
            else:
                utt_ids.append(utt_id)
        recent_context.append(sent)
        if utt_id != ans_utt_id and sent_speaker == ans_speaker:
            flag = True
    recent_context.reverse()

    if len(speakers) < 3:
        flag = False
    if any([ c == 0 for c in role_counts]):
        flag = False
    if answer["role"] == "mod" and answer["id"].split("_")[1] != "0":
        flag = False
    if other_roles:
        flag = False

    conv_context_text = "\n".join([sent["speaker"] + ": " + sent["text"] for sent in recent_context])
    if len(conv_context_text.split(" ")) > 2048:
        flag = False

    return flag, recent_context


def generate_features(args, convs, split_label, sample_size = 60):

    debates = []
    all_sampled_instances = []
    all_valid_instances = []

    for index, convo in convs:
        debate = {"title": convo.meta['title'],
                  "results": convo.meta['results'],
                  "winner": convo.meta['winner'],
                  "id": index}
        meta = {"title": convo.meta['title'],
                "results": convo.meta['results'],
                "winner": convo.meta['winner'],
                "id": index}
        speakers = {}
        dialogue = []
        instance_candidates = []
        sampled_instances = []
        utt_counts = 0
        context_length = args.max_utt_num
        for utt in convo.iter_utterances():
            if utt.speaker_.id not in speakers and utt.meta['speakertype'] in ["for", "against", "mod"]:
                speaker = utt.speaker_
                role = utt.meta['speakertype']

                speakers[utt.speaker_.id] = {
                    "name":speaker.id,
                    "bio":speaker.meta["bio"],
                    "bio_short": speaker.meta["bio_short"],
                    "role": role,
                    "statement": None
                }
                if role in ["for", "against"]:
                    statements = [ (uid, sutt) for uid, sutt in speaker.utterances.items() if sutt.meta['segment'] == 0 ]
                    statement = max(statements, key = lambda x: len(x[1].text.split(" ")))[1].text
                    speakers[utt.speaker_.id]["statement"] = statement
            if utt.meta['segment'] == 1:
                sents = sent_tokenize(utt.text)
                for j, sent in enumerate(sents):
                    d = {
                        "id": str(utt.id) + "_" + str(j),
                        "speaker": utt.speaker_.id,
                        "role": utt.meta['speakertype'],
                        "segment": utt.meta['segment'],
                        "text": sent,
                        "non-text": utt.meta["nontext"],
                        "dialogue act": "",
                        "motivation": "",
                        "replied to": ""
                    }
                    dialogue.append(d)
                    if utt_counts >= context_length:
                        validation_flag, recent_context = validate_instance(d, dialogue, context_length)
                        if validation_flag:
                            instance = {
                                "answer": d,
                                "context": recent_context,
                                "meta": meta,
                                "speakers": speakers
                            }
                            instance_candidates.append(instance)
                utt_counts += 1
        sample_size = min(sample_size, len(instance_candidates))
        for role, ratio in role_ratio.items():
            number = int(sample_size * ratio)
            filtered_candidates = [cand for cand in instance_candidates if cand["answer"]["role"] == role]
            if len(filtered_candidates) > number:
                filtered_candidates = random.sample(filtered_candidates, number)
            sampled_instances.extend(filtered_candidates)
        debate["speakers"] = speakers
        debate["dialogue"] = dialogue
        debates.append(debate)
        all_sampled_instances.extend(sampled_instances)
        all_valid_instances.extend(instance_candidates)

    return all_sampled_instances, all_valid_instances

def generate_speakers_string(speakers):
    strings_list = []
    for name, info in speakers.items():
        bio = info["bio_short"]
        if not bio:
            bio = "unknown"
        speaker_string = name + f"<{info['role']}>" + ": " + bio
        strings_list.append(speaker_string)
    return "\n".join(strings_list)

def generate_dialogue_history_string(dialogue_history):
    strings_list = []
    for utt in dialogue_history:
        try:
            utt_string = utt["speaker"] + f" (role: {utt['role']}): " + utt["text"]
            strings_list.append(utt_string)
        except Exception as e:
            print(e)
    return "\n".join(strings_list)


def process_and_save(data, output_path, split_label="test"):
    instances = []
    for datum in data:
        instance ={
            "debate_id": datum["meta"]["id"],
            "utterance_id": datum["answer"]["id"],
            "split": split_label,
            "title": datum["meta"]["title"],
            "speakers": generate_speakers_string(datum["speakers"]),
            "dialogue_history": generate_dialogue_history_string(datum["context"]),
            "debate_result": datum["meta"]["results"],
            "ans_speaker": datum["answer"]["speaker"],
            "ans_role": datum["answer"]["role"],
            "ans_text": datum["answer"]["text"],
        }
        instances.append(instance)
    df = pd.DataFrame(instances)
    df.to_csv(output_path, index=False)
    return instances

def main():
    args = get_args()
    corpus = Corpus(filename=download("iq2-corpus"))
    test_debates, train_debates = test_train_split(corpus, 30)
    sampled_test_data, full_test_debates = generate_features(args, test_debates, split_label="test")
    sampled_train_data, full_train_debates  = generate_features(args, train_debates, split_label="train")
    process_and_save(sampled_test_data, "data/archived/isd_sampled_test.csv")
    process_and_save(full_test_debates, "data/archived/isd_full_test.csv")
    process_and_save(sampled_train_data, "data/archived/isd_sampled_train.csv")
    process_and_save(full_train_debates, "data/archived/isd_full_train.csv")
    print("test debates ids: ")
    print(", ".join([d["id"] for d in test_debates]))
    print("train debates ids: ")
    print(", ".join([d["id"] for d in train_debates]))
    print("finish!")



if __name__ == '__main__':

    main()
    # debates = generate_features(args)
    # save_data(debates, args.data_dir)

