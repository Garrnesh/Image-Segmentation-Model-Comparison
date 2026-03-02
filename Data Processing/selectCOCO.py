import pickle
import json
import random

with open('./dataVal.pkl', 'rb') as file:
    data = pickle.load(file)

print([len(d) for d in data])
newImages = []
seen = []
for i, d in enumerate(data):
    new = []
    random.shuffle(d)
    if i==1:
        limit = 105
    elif i==2:
        limit = 92
    elif i==3:
        limit = 102
    else:
        limit = 75
    
    while len(new)<limit:
        val = d.pop(0)
        if val not in seen:
            new.append(val)
        seen.append(val)
    newImages.append(new)

classes = ['car', 'bus', 'motorcycle', 'truck', 'person']
finalData = {}
print([len(img) for img in newImages])
for i, c in enumerate(classes):
    finalData[c] = newImages[i]

with open('./valtest75COCOClassSplit.json', 'w') as file:
    json.dump(finalData, file)