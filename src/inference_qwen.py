from transformers import AutoModelForCausalLM, AutoTokenizer
from constant import MODEL_PATH, PROMPT_Qwen, TEST_DATA_PATH, RESULT_DIR, PROMPT_Qwen_based_kb
import json
from tqdm import tqdm
from utils import get_ans_postprocess, get_relate_knowledge, post_process_kb, get_acc, get_dual_relate_kb
device = "cuda" # the device to load the model onto
question_list = []
choice_list = []
answer_list = []
output_list = []
rag_list = []
raw_output_list = []
right_samples = []
error_samples = []


model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH + "Qwen1.5-7B-Chat",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH + "Qwen1.5-7B-Chat", trust_remote_code=True)


def get_dataset():
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as R:
        datasets = json.load(R)
    return datasets


if __name__ == "__main__":
    from pandas import DataFrame

    for i, itm in tqdm(enumerate(get_dataset())):
        kb = get_dual_relate_kb(str(itm["statement"] + "\n" + str(itm["option_list"])))
        input_str = str(PROMPT_Qwen_based_kb.format(question=itm["statement"], choice=itm["option_list"],
                                                    kb="\t".join(kb)))
        input_tensor = tokenizer(input_str, return_tensors="pt")
        question_list.append(itm["statement"])
        choice_list.append(itm["option_list"])
        answer_list.append(itm["answer"])
        rag_list.append(kb)

        prompt = input_str
        messages = [
            {"role": "system", "content": PROMPT_Qwen_based_kb},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        raw_output_list.append(response)
        response = get_ans_postprocess(response)
        output_list.append(response)
        if "".join(itm["answer"]) == response:
            right_samples.append({
                "answer": itm["answer"],
                "question": itm["statement"],
                "choice": itm["option_list"]
            })
        else:
            error_samples.append({
                "answer": itm["answer"],
                "question": itm["statement"],
                "choice": itm["option_list"]
            })

        # print(response)
    file_path = RESULT_DIR + 'cail/Qwen1.5-7B-Chat_zeroshot_withrag.xlsx'
    data = {'question': question_list, 'choice': choice_list,
            'ans': answer_list, 'output': output_list, 'raw_output_list': raw_output_list, 'rag': rag_list}
    df = DataFrame(data)
    df.to_excel(file_path)

    with open("right_data.json", 'w', encoding='utf-8') as W:
        json.dump(right_samples, W, ensure_ascii=False, indent=4)

    with open("right_data.json", 'w', encoding='utf-8') as R:
        json.dump(right_samples, R, ensure_ascii=False, indent=4)

    acc = get_acc('cail/Qwen1.5-7B-Chat_zeroshot_withrag.xlsx')
    print(acc)
    """
    # Qwen1.5-7B-Chat acc 0.2661064425770308
    # post process 0.29411764705882354
    # rag 0.2735760971055089 
    # rag_ 0.2857142857142857
    # rag_dual_post_ 0.26517273576097106
    # rag_dual_post_ 0.26704014939309056
    """
