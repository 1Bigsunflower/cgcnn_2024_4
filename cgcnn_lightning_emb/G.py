import json

data = {}
for i in range(1, 104):
    data[str(i)] = [i]

with open('atom.json', 'w') as json_file:
    json.dump(data, json_file)
