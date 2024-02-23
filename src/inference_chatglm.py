from transformers import AutoModel, AutoTokenizer
from constant import MODEL_PATH, TEST_DATA_PATH, RESULT_DIR, PROMPT_Qwen_based_kb, PROMPT_BASE
import json
from tqdm import tqdm
from utils import get_ans_postprocess, get_relate_knowledge

device = "cuda"  # the device to load the model onto
question_list = []
choice_list = []
answer_list = []
output_list = []
rag_list = []

model = AutoModel.from_pretrained(
    MODEL_PATH + "THUDM-chatglm3-6b",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH + "THUDM-chatglm3-6b", trust_remote_code=True)


def get_dataset():
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as R:
        datasets = json.load(R)
    return datasets


if __name__ == "__main__":
    from pandas import DataFrame
    """
    for i, itm in tqdm(enumerate(get_dataset())):
        kb = get_relate_knowledge(str(itm["statement"] + "\n" + str(itm["option_list"])))
        input_str = str(PROMPT_BASE.format(question=itm["statement"], choice=itm["option_list"]))
        input_tensor = tokenizer(input_str, return_tensors="pt")
        question_list.append(itm["statement"])
        choice_list.append(itm["option_list"])
        answer_list.append(itm["answer"])
        # rag_list.append(kb)
        model = model.eval()
        response, history = model.chat(tokenizer, input_str, history=[])
        output_list.append(response)

    file_path = RESULT_DIR + 'cail/THUDM-chatglm3-6b_zeroshot_withoutrag.xlsx'
    data = {'question': question_list, 'choice': choice_list,
            'ans': answer_list, 'output': output_list}
    df = DataFrame(data)
    df.to_excel(file_path)

    """
    import pandas as pd

    df = pd.read_excel(RESULT_DIR + 'cail/THUDM-chatglm3-6b_zeroshot_withoutrag.xlsx')
    ans = df["ans"].to_list()
    output = df["output"].to_list()

    assert len(ans) == len(output)
    cnt = 0
    for i, itm in tqdm(enumerate(ans)):
        ans_ = "".join(eval(ans[i]))
        output_ = get_ans_postprocess(str(output[i]))
        # \str(output[i])

        if ans_ == output_:
            cnt += 1
    print(float(cnt/len(ans)))
    # THUDM-chatglm3-6b_zeroshot_withoutrag acc 0.1568627450980392
    # post process 0.29411764705882354
