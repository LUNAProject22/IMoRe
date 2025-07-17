import re
import json
import sys
import json
from nltk.stem import WordNetLemmatizer
import json
import sys
import Constants


lemmatizer = WordNetLemmatizer()


def create_inputs(dataset_type, output_path):
    """Process the dataset (train, test, or val) and save results to a JSON file."""

    def find_all_nums(strings):

        nums = []
        args = []
        for s in strings:
            if "[" in s and "]" in s:
                nums.append(int(s[1:-1]))

            if "[" in s and "]" not in s:
                x = [int(re.sub(r"\D", "", s))]
                nums.append(len((x)))
                args.append(x[0])
            elif "]" in s and "[" not in s:
                x = [int(re.sub(r"\D", "", s))]
                nums.append(len((x)))
                args.append(x[0])
            else:
                continue
        # print("nums: ", nums)
        return nums, args

    dataset_file = f"./IPGRM_formatted_data/BABELQA_{dataset_type}.json"
    print(f"Loading {dataset_file}")
    with open(dataset_file) as f:
        data = json.load(f)

    results = []
    for idx, entry in enumerate(data):
        # for prog in entry[2]:
        programs = entry[2]
        rounds = []
        depth = {}
        cur_depth = 0
        tmp = []
        connection = []
        inputs = []
        returns = []
        tmp_connection = []
        for i, program in enumerate(programs):
            if isinstance(program, list):
                _, func, args = Constants.parse_program(program[1])
                returns.append(program[0])
            else:
                _, func, args = Constants.parse_program(program)
            try:
                if func == "query_body_part":
                    inputs.append([func, None, None, None, None, None, None, None])

                elif func == "query_action":
                    inputs.append([func, None, None, None, None, None, None, None])

                elif func == "query_direction":
                    inputs.append([func, None, None, None, None, None, None, None])

                elif func == "filter_action":
                    inputs.append([func, args[1], None, None, None, None, None, None])

                elif func == "filter_direction":
                    inputs.append([func, None, args[1], None, None, None, None, None])

                elif func == "filter_body_part":
                    # query = func
                    # filter_type = query.replace("filter_", "")
                    inputs.append([func, None, None, args[1], None, None, None, None])

                elif func == "relate":
                    inputs.append([func, None, None, None, args[1], None, None, None])

                elif func == "scene":
                    inputs.append([func, None, None, None, None, None, None, None])

                elif func == "intersect":
                    inputs.append([func, None, None, None, None, None, None, None])

                else:
                    raise ValueError("unknown function {}".format(func))
            except Exception:
                print("exception :", program)
                inputs.append([func, None, None, None, None, None, None, None])

            assert len(inputs[-1]) == 8
            c, a = find_all_nums(args)
            if len(c) == 0:
                tmp.append(program)
                depth[i] = cur_depth
                tmp_connection.append([i, i])
            elif len(c) == 2:
                tmp.append(program)
                depth[i] = cur_depth
                tmp_connection.append([i, a])

        connection.extend(tmp_connection)
        # connection.append(tmp_connection) # original
        cur_depth += 1
        rounds.append(tmp)

        while len(depth) < len(programs):
            tmp = []
            tmp_depth = {}
            tmp_connection = []
            for i, program in enumerate(programs):
                if i in depth:
                    continue
                else:
                    if isinstance(program, list):
                        _, func, args = Constants.parse_program(program[1])
                    else:
                        _, func, args = Constants.parse_program(program)
                    c, _ = find_all_nums(args)
                    # if len(c)==2:
                    #     continue
                    if all([_ in depth for _ in c]) and len(c) != 2:
                        tmp.append(program)
                        tmp_depth[i] = cur_depth
                        for r in c:
                            if r > i:
                                r = i - 1
                            tmp_connection.append([i, r])
                    # else:
                    #     continue

            if len(tmp_depth) == 0 and len(tmp) == 0 and len(tmp_connection) == 0:
                break
            else:
                connection.extend(tmp_connection)
                rounds.append(tmp)
                cur_depth += 1
                depth.update(tmp_depth)

        connection_sorted = sorted(connection, key=lambda x: x[0])

        results.append(
            [
                entry[0],
                entry[1],
                returns,
                inputs,
                connection_sorted,
                entry[-2],
                entry[-1],
            ]
        )
        sys.stdout.write(f"Finished {idx+1}/{len(data)} \r")

    # Write the output to JSON
    output_file = f"{output_path}/BABELQA_{dataset_type}_inputs.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Done! Results saved to {output_file}")


def process_all_splits():
    output_path = "./IPGRM_formatted_data"
    splits = ["train", "test", "val"]

    for split in splits:
        create_inputs(split, output_path)


if __name__ == "__main__":
    process_all_splits()
