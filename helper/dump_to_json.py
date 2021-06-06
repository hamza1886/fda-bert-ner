import json


def dump_to_json(json_data, filename, indent=False):
    data = json.dumps(json_data, indent=indent)
    with open(filename, 'w+') as f:
        f.write(data)
