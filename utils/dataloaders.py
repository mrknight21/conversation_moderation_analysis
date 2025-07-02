
import json


def get_episode_meta(episode):
    meta_json_path = episode.replace(".xlsx", "_meta.json")
    meta = None
    with open(meta_json_path) as f:
        metas = json.load(f)
        if len(metas.values()) > 0:
            meta = list(metas.values())[0]
        meta["topic"] = " ".join(meta["topic"].split("_")[2:])
    return meta


def write_txt(lines:list, filepath:str):
    with open(filepath, 'w') as f:
        f.write("\n".join(lines))


def load_json_data(path):
    with open(path) as f:
        json_objs = json.load(f)
        return json_objs


def read_jsonl_to_json(filepath):
    """
    Reads a .jsonl (JSON Lines) file and returns a list of JSON objects.

    Args:
        filepath (str): The path to the .jsonl file.

    Returns:
        list: A list of dictionaries representing the JSON objects in the file.
    """
    json_objects = []
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                json_objects.append(json.loads(line.strip()))
        return json_objects
    except Exception as e:
        print(f"Error reading the file: {e}")
        return []

