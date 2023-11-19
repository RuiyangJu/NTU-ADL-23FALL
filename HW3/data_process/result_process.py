import json

with open('../data/private_test.json', 'r', encoding='utf-8') as f:
    data_test = json.load(f)
for key in data_test:
    if key["instruction"]:
        del key["instruction"]
print(data_test)
'''
with open('data/id.json', 'w', encoding='utf-8') as f:
    #json.dump(data_test, f, ensure_ascii=False, indent=2)
'''

with open('../data/generated_predictions.jsonl', 'r', encoding='utf-8') as f:
   json_list = list(f)
data_generated = []
for json_str in json_list:
    data_generated.append(json.loads(json_str))
for key in data_generated:
    if key["label"] == "":
        del key["label"]
    key["output"] = key.pop("predict")
print(data_generated)
'''
with open('data/output.json', 'w', encoding='utf-8') as f:
    json.dump(data_generated, f, ensure_ascii=False, indent=2)
'''

for i in range(len(data_test)):
    data_test[i].update(data_generated[i])
    data_result = json.dumps(data_test, indent=2, separators=(',', ': '), ensure_ascii = False)
with open('../prediction.json', 'w', encoding = 'utf-8') as f:
    f.write(data_result)
