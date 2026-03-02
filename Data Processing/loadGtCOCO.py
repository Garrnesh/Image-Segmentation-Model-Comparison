from pycocotools.coco import COCO
import os
import json
import copy

annTrain = './annotations/instances_train2017.json'
valTrain = './annotations/instances_val2017.json'

cocoTrain = COCO(annTrain)
cocoVal = COCO(valTrain)

classes = ['car', 'bus', 'motorcycle', 'truck', 'person']
classIds = [cocoVal.getCatIds(catNms=[cat])[0] for cat in classes]

gt_data = {}

with open('./valtest75COCOClassSplit.json', 'r') as file:
    data = json.load(file)

for key, val in data.items():
    store = {}
    for img in val:
        inner_store = {}
        annIds = cocoVal.getAnnIds(imgIds=img)
        anns = cocoVal.loadAnns(annIds)

        for ann in anns:
            catId = ann['category_id']
            if catId not in classIds:
                continue
            index = classIds.index(catId)
            curClass = classes[index]

            if curClass not in inner_store:
                inner_store[curClass] = []

            inner_store[curClass].append(ann["id"])

        store[img] = inner_store
    gt_data[key] = store

with open('valtest75COCOGt.json', 'w') as file:
    json.dump(gt_data, file, indent=4)