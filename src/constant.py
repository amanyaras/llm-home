PROMPT = "现在你是一名优秀的法律从业者，请专业、详尽的回答用户提问。\n" \
                "注意：你需要遵循以下规则：\n" \
                "1、你只能输出ABCD中的一个或者多个。\n" \
                "2、ABCD代表了对应的句子\n"\
                "在符合上述规则的基础上你需要做的是：根据问题和选项输出对应的答案。\n" \
                "当问题中只出现品牌或者只出现型号的时候，品牌或型号对应已知信息中每一条信息中的品牌或型号其中之一就算包含。" \
                "问题是：{question}。\n选项是：{choice}。你的答案是:"

TEST_DATA_PATH = "/home/zhangyh/projs/llm-home/cail_data/test_data.json"
RESULT_DIR = "/home/zhangyh/projs/llm-home/results/"
MODEL_PATH = "/home/zhangyh/models/"
LOG_DIR = "/home/zhangyh/projs/llm-home/logs/"
