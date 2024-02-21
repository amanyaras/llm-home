import json
from tqdm import tqdm
from constant import PROMPT, TEST_DATA_PATH, RESULT_DIR, MODEL_PATH, LOG_DIR
from transformers import AutoTokenizer, LlamaForCausalLM
from fastchat.utils import build_logger
LOG_DIR = LOG_DIR
logger = build_logger("ziya2_13b", "ziya2_13b_0")


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
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH + "ziya2-13b-chat", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH + "ziya2-13b-chat", trust_remote_code=True,
                                              use_fast=False)
    for i, itm in tqdm(enumerate(get_dataset())):
        input_str = str(PROMPT.format(question=itm["statement"], choice=itm["option_list"]))
        input_tensor = tokenizer(input_str, return_tensors="pt")
        question_list.append(itm["statement"])
        choice_list.append(itm["option_list"])
        answer_list.append(itm["answer"])
        output = model.generate(**input_tensor)
        pred = tokenizer.batch_decode(output, skip_special_tokens=True)
        output_list.append(pred)

    file_path = RESULT_DIR + 'cail/ziya2_13b_zeroshot.xlsx'
    data = {'question': question_list, 'choice': choice_list,
            'ans': answer_list, 'output': output_list}
    df = DataFrame(data)
    df.to_excel(file_path)


