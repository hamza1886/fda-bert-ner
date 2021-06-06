import json
import os


def append_to_json(key, value, path):
    content = {}

    # read file content, if file is not empty
    if os.stat(path).st_size != 0:
        with open(path, 'r') as f:
            content = json.load(f)

    # dump file content to json
    with open(path, 'w') as f:
        content[key] = value
        json.dump(content, f)
