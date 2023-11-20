import json
import argparse

def parse_args():
    parse = argparse.ArgumentParser(description='Data Preprocess')
    parse.add_argument('--input', type=str, default=None, help='Path to input data.')
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for key in data:
        if key["id"]:
            del key["id"]
        key["input"] = ""
        key["output"] = ""

    with open('data/processed_input.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Data Preprocessing is Finished!")
