import json


def convert_to_list(data):
    output = []

    for question_id, content in data.items():

        question = content["question"]
        answer = content["answer"]
        babel_id = content["babel_id"]

        # Process the program
        program_steps = content["program"]
        program_list = []
        for i, step in enumerate(program_steps):
            func = step["function"]
            inputs = step["inputs"]
            value_inputs = step["value_inputs"]

            # Format the inputs and function call
            if inputs:
                if len(inputs) == 1:
                    input_str = f"[{inputs[0]}]"
                elif len(inputs) > 1:
                    input_str = f"{inputs}"
                else:
                    print("Error!!")

            else:
                input_str = f"[{i}]"

            if value_inputs:
                func_str = f"{func}({input_str}, {', '.join(value_inputs)})"
            else:
                func_str = f"{func}({input_str})"

            if func.startswith("query") or func.startswith("exist"):
                func_str = f"?={func_str}"
            else:
                func_str = f"[{i}]={func_str}"

            if func_str == '[0]=scene([0])':
                func_str = '[0]=scene(motion_seq)'
            program_list.append(func_str)

        output.append(
            [
                question_id,  # question_id
                question,  # question
                program_list,  # program list
                babel_id,  # babel_id
                answer,  # answer
            ]
        )

    return output


# Convert and print the result
with open("./BABEL_QA_data/questions.json", "r") as f:
    data = json.load(f)

result = convert_to_list(data)
# print(result[:2])


# Load train/test/val IDs
with open("./BABEL_QA_data/split_question_ids.json", "r") as f:
    data = json.load(f)  # This will contain the 'train', 'test', and 'val' lists


# Function to filter results based on train, test, and val keys
def filter_results(data_key):
    return [entry for entry in result if entry[0] in data[data_key]]


# Filter results for train, test, and val
train_results = filter_results("train")
test_results = filter_results("test")
val_results = filter_results("val")

# Save filtered results to separate JSON files
with open("./IPGRM_formatted_data/BABELQA_train.json", "w") as train_file:
    json.dump(train_results, train_file, indent=4)

with open("./IPGRM_formatted_data/BABELQA_test.json", "w") as test_file:
    json.dump(test_results, test_file, indent=4)

with open("./IPGRM_formatted_data/BABELQA_val.json", "w") as val_file:
    json.dump(val_results, val_file, indent=4)

print("Files written to train_results.json, test_results.json, val_results.json!")

# Save all data
output_file = "./IPGRM_formatted_data/BABELQA.json"
with open(output_file, "w") as f:
    json.dump(result, f, indent=2)

print(f"Convetered BABELQA to GQA format {output_file}")
