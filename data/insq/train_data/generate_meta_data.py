import glob
import json
import pandas as pd

mode = "train"
INPUT_FOLDER = f"./"
OUTPUT_FOLDER = f"./"

# {
#     "anh": {
#         "topic": "8_2228_Guns_Reduce_Crime",
#         "impression": {
#             "pre-annotation": {
#                 "attitude": 2,
#                 "understanding": 3
#             },
#             "post_annotation": {
#                 "attitude": 3,
#                 "understanding": 4,
#                 "constructive": 5,
#                 "knowledgeable": 4,
#                 "emotional": 2,
#                 "helpful": 5,
#                 "bias": 3
#             }
#         },
#         "speakers": [
#             "0 (Unknown)",
#             "1 (Self)",
#             "2 (Everyone)",
#             "3 (Audience)",
#             "4 (John R. Lott- for)",
#             "5 (Stephen Halbrook- for)",
#             "6 (Gary Kleck- for)",
#             "7 (R. Gil Kerlikowske- against)",
#             "8 (John J. Donohue- against)",
#             "9 (Paul Helmke- against)",
#             "10 (Support team)",
#             "11 (Against team)",
#             "12 (All speakers)"
#         ],
#         "annotator": "anh",
#         "error_log": [
#
#         ]
#     },

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