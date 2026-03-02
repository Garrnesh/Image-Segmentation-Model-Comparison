import os
import json
import copy
import random

training_folder = './gtFine_trainvaltest/gtFine/train'
val_folder = './gtFine_trainvaltest/gtFine/val'
classes = ['car', 'truck', 'bus', 'motorcycle', 'person']
images = {
    c: [] for c in classes
}
final_data = {}
shuffle_images = []

for folderName in os.listdir(val_folder):
    internalFolder = os.path.join(val_folder, folderName)
    if not os.path.isdir(internalFolder):
        continue
    for fileName in os.listdir(internalFolder):
        if fileName[0:2]!='._' and fileName.endswith('.json'):
            shuffle_images.append(os.path.join(internalFolder, fileName))

random.shuffle(shuffle_images)

count = 0
for jsonPath in shuffle_images:
    with open(jsonPath, 'r') as file:
        data = json.load(file)
    new_data = {'height': data['imgHeight'], 'width': data['imgWidth']}
    track_data = {}
    for obj in data['objects']:
        if obj['label'] == 'rider':
            if 'person' not in new_data:
                new_data['person'] = []
            new_data['person'].append(obj["polygon"])
            if 'person' not in track_data:
                track_data['person'] = 0
            track_data['person'] += 1
        elif obj['label'] in classes:
            if obj['label'] not in new_data:
                new_data[obj['label']] = []
            new_data[obj['label']].append(obj['polygon'])
            if obj['label'] not in track_data:
                track_data[obj['label']] = 0
            track_data[obj['label']] += 1

    imageId = '_'.join(jsonPath.split('/')[-1].split('_')[0:3])
    final_data[imageId] = new_data

    if len(track_data)<2:
        continue
    shuffle_classes = copy.deepcopy(classes)
    random.shuffle(shuffle_classes)
    for c in shuffle_classes:
        if len(images[c])>=75:
            continue
        if c not in track_data:
            continue
        if track_data[c]>=1:
            images[c].append(imageId)
            break

with open('valtestCity75.json', 'w') as file:
    json.dump(final_data, file, indent=4)

with open('valtestClassSplit75.json', 'w') as file:
    json.dump(images, file, indent=4)

with open('./valtestCity75.json', 'r') as file:
    data = json.load(file)

with open('./valtestClassSplit75.json', 'r') as file:
    splitData = json.load(file)

final_list = []
for key, val in splitData.items():
    print(key, len(val))
    final_list += val

print(len(final_list), len(set(final_list)))

print(len(data))