import subprocess
from carate.utils.file_utils import save_json_to_file

def convert_py_to_json(file_name:str, out_name:str)->dict: 
    
    json_dict = {}
    lines = read_file(file_name)

    for line in lines: 
        tmp = line.split(" ")
        if tmp[2] == 'None' or tmp[2] == "None": 
            json_dict[tmp[0]] = None
        else:
            json_dict[tmp[0]] = tmp[2]
    
    save_json_to_file(json_dict, file_name=out_name)
    return json_dict

def read_file(file_name:str)->list[str]:
    subprocess.run(["black", file_name], capture_output=True) #format file to avoid misunderstandings 
    with open(file_name) as file: 
        raw = file.readlines()
    result = sanitize_raw_py(raw)
    return result

def sanitize_raw_py(raw_input:list[str])->list[str]:
    result = []
    for line in raw_input: 
        if line == "\n": 
            continue
        else: 
            result.append(line.replace("\n", "").replace('"', '').replace("'", ""))
    return result
