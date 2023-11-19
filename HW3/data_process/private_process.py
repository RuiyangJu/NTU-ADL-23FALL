import json

with open('../data/private_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for key in data:
    if key["id"]:
        del key["id"]
    key["input"] = ""
    key["output"] = ""
print(data)

with open('../data/processed_private.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
