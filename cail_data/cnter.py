import os
import json


if __name__ == "__main__":
    with open("./train.json", 'r', encoding='utf-8') as F:
        data = json.load(F)

    print(len(data))