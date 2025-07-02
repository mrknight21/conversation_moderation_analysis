import glob
import json
import pandas as pd

mode = "train"
INPUT_FOLDER = f"./"
OUTPUT_FOLDER = f"./"

def main():
    episodes = glob.glob(INPUT_FOLDER + "/*.xlsx")
    for e in episodes:
        complimentary_speaker_tags = ["(Support team)", "(Against team)", "(All speakers)"]
        topic = e.split("/")[-1].replace(".xlsx", "")
        labels = pd.read_excel(e, sheet_name='labels')
        speaker_labels = labels.speakers.tolist()
        speaker_index = len(speaker_labels)
        for t in complimentary_speaker_tags:
            if not any([t in l for l in speaker_labels]):
                new_tag = f"{speaker_index} {t}"
                speaker_labels.append(new_tag)
                speaker_index += 1
        meta = {"default":{"topic": topic, "speakers":speaker_labels}}
        meta_file_path = OUTPUT_FOLDER + topic + "_meta.json"
        with open(meta_file_path, 'w') as fp:
            json.dump(meta, fp)



if __name__ == "__main__":
    main()