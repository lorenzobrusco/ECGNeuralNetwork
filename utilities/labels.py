import json

labels_json = '{ ".": "NOR", "N": "NOR", "V": "PVC", "/": "PAB", "L": "LBB", "R": "RBB", "A": "APC", "!": "VFW", "E": "VEB" }'
labels_to_float = '{ "NOR": "0", "PVC" : "1", "PAB": "2", "LBB": "3", "RBB": "4", "APC": "5", "VFW": "6", "VEB": "7" }'
float_to_labels = '{ "0": "NOR", "1" : "PVC", "2": "PAB", "3": "LBB", "4": "RBB", "5": "APC", "6": "VFW", "7": "VEB" }'
labels = json.loads(labels_to_float)
revert_labels = json.loads(float_to_labels)
original_labels = json.loads(labels_json)
