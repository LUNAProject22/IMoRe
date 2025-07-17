# sorce codes from https://github.com/wenhuchen/Meta-Module-Network

import os
import json
from multiprocessing import Pool
import multiprocessing
from nltk.tokenize import word_tokenize


def split(string):
    output = []
    buf_str = ""
    for s in string:
        if s == "(":
            string = buf_str.strip()
            if string:
                output.append(string)
            output.append("(")
            buf_str = ""
        elif s == ")":
            string = buf_str.strip()
            if string:
                output.append(string)
            output.append(")")
            buf_str = ""
        elif s == ",":
            string = buf_str.strip()
            if string:
                output.append(string)
            output.append(",")
            buf_str = ""
        else:
            buf_str += s
    return output


def generate_pairs(entry):
    if entry[2]:
        output = []
        for r in entry[2]:
            _, p = r.split("=")
            sub_p = split(p)
            output.extend(sub_p)
            output.append(";")
        del output[-1]
    else:
        output = []

    question = word_tokenize(entry[1])
    return (question, output, entry[0], entry[-2], entry[-1])


# Define file paths for train, val, and test datasets
datasets = {
    "train": "./IPGRM_formatted_data/BABELQA_train.json",
    "val": "./IPGRM_formatted_data/BABELQA_val.json",
    "test": "./IPGRM_formatted_data/BABELQA_test.json",
}

for dataset_type, filename in datasets.items():
    if os.path.exists(filename):
        print(f"Processing {dataset_type} dataset from {filename}...")

        with open(filename) as f:
            data = json.load(f)
            print(f"Total {len(data)} programs in {dataset_type} set.")

            cores = multiprocessing.cpu_count()
            print(f"Using parallel computing with {cores} cores.")
            pool = Pool(cores)

            r = pool.map(generate_pairs, data)
            print(f"Processed {len(r)} records for {dataset_type} set.")

            pool.close()
            pool.join()

            output_file = f"./IPGRM_formatted_data/BABELQA_{dataset_type}_pairs.json"
            with open(output_file, "w") as f:
                json.dump(r, f)

            print(f"Created {dataset_type} pairs file: {output_file}\n")
    else:
        print(f"File {filename} does not exist. Skipping {dataset_type} dataset.")
