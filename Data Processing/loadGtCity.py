import json
import copy

with open('./valtestCity75.json', 'r') as file:
    data = json.load(file)

with open('./valtestClassSplit75.json', 'r') as file:
    splitData = json.load(file)

city_gt = {}

for key, val in splitData.items():
    store = {}
    for img in val:
        details = data[img]
        addData = copy.deepcopy(details)
        addData.pop("height")
        addData.pop("width")

        store[img] = addData

    city_gt[key] = store

with open('./valtest75CityGt.json', 'w') as file:
    json.dump(city_gt, file, indent=4)