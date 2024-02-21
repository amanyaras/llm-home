import json
from tqdm import tqdm
from constant import PROMPT, TEST_DATA_PATH, RESULT_DIR, MODEL_PATH, LOG_DIR, PROMPT_Qwen
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.utils import build_logger
LOG_DIR = LOG_DIR
# logger = build_logger("ziya2_13b", "ziya2_13b_0")

question_list = []
choice_list = []
answer_list = []
output_list = []


def get_dataset():
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as R:
        datasets = json.load(R)
    return datasets


if __name__ == "__main__":

    from pandas import DataFrame
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH + "Qwen-14B-Chat-Int8", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH + "Qwen-14B-Chat-Int8", trust_remote_code=True,
                                              use_fast=False)
    for i, itm in tqdm(enumerate(get_dataset())):
        input_str = str(PROMPT_Qwen.format(question=itm["statement"], choice=itm["option_list"]))
        input_tensor = tokenizer(input_str, return_tensors="pt")
        question_list.append(itm["statement"])
        choice_list.append(itm["option_list"])
        answer_list.append(itm["answer"])
        output = model.generate(**input_tensor)
        pred = tokenizer.batch_decode(output, skip_special_tokens=True)
        output_list.append(pred)
        print(pred)
        break

    file_path = RESULT_DIR + 'cail/Qwen-14B-Chat-Int8_zeroshot.xlsx'
    data = {'question': question_list, 'choice': choice_list,
            'ans': answer_list, 'output': output_list}
    df = DataFrame(data)
    df.to_excel(file_path)

    """
    import pandas as pd

    df = pd.read_excel(RESULT_DIR + 'cail/ziya2_13b_zeroshot.xlsx')
    ans = df["ans"].to_list()
    output = df["output"].to_list()

    assert len(ans) == len(output)
    cnt = 0
    for i, itm in tqdm(enumerate(ans)):
        ans_ = "".join(eval(ans[i]))
        output_ = eval(output[i])[0].split("你的答案是:")[1]

        if ans_ == output_:
            cnt += 1
    print(float(cnt/len(ans)))
    # ziya acc 0.15779645191409897
    """
