from transformers import AutoModelForCausalLM, AutoTokenizer
from constant import MODEL_PATH, PROMPT_Qwen, TEST_DATA_PATH, RESULT_DIR
import json
from tqdm import tqdm
device = "cuda" # the device to load the model onto
question_list = []
choice_list = []
answer_list = []
output_list = []

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
        input_str = str(PROMPT_Qwen.format(question=itm["statement"], choice=itm["option_list"]))
        input_tensor = tokenizer(input_str, return_tensors="pt")
        question_list.append(itm["statement"])
        choice_list.append(itm["option_list"])
        answer_list.append(itm["answer"])

        prompt = input_str
        messages = [
            {"role": "system", "content": PROMPT_Qwen},
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

        output_list.append(response)
        # print(response)
    file_path = RESULT_DIR + 'cail/Qwen1.5-7B-Chat.xlsx'
    data = {'question': question_list, 'choice': choice_list,
            'ans': answer_list, 'output': output_list}
    df = DataFrame(data)
    df.to_excel(file_path)

