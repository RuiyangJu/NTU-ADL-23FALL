import json
import argparse

def parse_args():
    parse = argparse.ArgumentParser(description='Data Postprocess')
    parse.add_argument('--original', type=str, default=None, help='Path to original data.')
    parse.add_argument('--generated', type=str, default=None, help='Path to generated data.')
    parse.add_argument('--output', type=str, default=None, help='Path to output data.')
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    with open(args.original, 'r', encoding='utf-8') as f:
        data_test = json.load(f)
    for key in data_test:
        if key["instruction"]:
            del key["instruction"]

    with open(args.generated, 'r', encoding='utf-8') as f:
        json_list = list(f)
    data_generated = []
    for json_str in json_list:
        data_generated.append(json.loads(json_str))
    for key in data_generated:
        if key["label"] == "":
            del key["label"]
        key["output"] = key.pop("predict")

    for i in range(len(data_test)):
        data_test[i].update(data_generated[i])
        data_result = json.dumps(data_test, indent=2, separators=(',', ': '), ensure_ascii = False)
    with open(args.output, 'w', encoding = 'utf-8') as f:
        f.write(data_result)
    print("Data Postprocessing is Finished!")
    print("The output.json is generated!")
