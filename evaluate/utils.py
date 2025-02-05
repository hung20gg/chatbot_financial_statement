import os 
import json


def append_jsonl_to_file(json_obj, file_path):
    with open(file_path, 'a') as f:
        json.dump(json_obj, f)
        f.write('\n')

def get_available_path(path):
    if not os.path.exists(path):
        return path
    else:
        i = 1
        file_type = path.split('.')[-1]
        path = path.replace('.' + file_type, '')

        while True:
            new_path = path + '_' + str(i) + '.' + file_type
            if not os.path.exists(new_path):
                return new_path
            i += 1

        