from pycocotools.coco import COCO
import os
import json
import pickle

annTrain = './annotations/instances_train2017.json'
valTrain = './annotations/instances_val2017.json'

cocoTrain = COCO(annTrain)
cocoVal = COCO(valTrain)

classes = ['car', 'bus', 'motorcycle', 'truck', 'person']
classIds = [cocoVal.getCatIds(catNms=[cat])[0] for cat in classes]
images = [[], [], [], [], []]

for i, cat in enumerate(classes):
    print('Performing checks for', cat)
    imgIds = cocoVal.getImgIds(catIds=classIds[i])

    for index, img in enumerate(imgIds):
        print(f'{index}/{len(imgIds)}')
        annIds = cocoVal.getAnnIds(imgIds=img)
        anns = cocoVal.loadAnns(annIds)

        catIdsInside = [ann['category_id'] for ann in anns]
        seen = []
        breakCounter = 0
        instanceCount = 0
        for catTest in catIdsInside:
            if catTest in classIds and catTest!=classIds[i] and catTest not in seen:
                seen.append(catTest)
                breakCounter+=1
            if catTest==classIds[i]:
                instanceCount+=1
        if breakCounter<1:
            continue

        images[i].append(img)

with open('./dataVal.pkl', 'wb') as file:
    pickle.dump(images, file)