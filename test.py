import json
from transformers import VivitModel, VivitConfig
from torch.nn import TransformerDecoderLayer

data_path = "outputs/synchronized_data.json"
with open(data_path, 'r') as f:
    data = json.load(f)

index = ""
while True:
    index = input("Enter number: ")
    if index == "q":
        break
    print(data[int(index)])