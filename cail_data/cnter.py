import os
import json
import random


if __name__ == "__main__":
    with open("./train.json", 'r', encoding='utf-8') as F:
        data = json.load(F)
    random.shuffle(data)
    with open("train_data.json", "w", encoding="utf-8") as W:
        json.dump(data[:20000], W, ensure_ascii=False, indent=4)

    with open("test_data.json", "w", encoding="utf-8") as E:
        json.dump(data[20000:-1], E, ensure_ascii=False, indent=4)
    print(len(data))