import json
from langchain.vectorstores.faiss import FAISS
from typing import List, Tuple
from langchain_core.documents import Document
from constant import TEST_DATA_PATH, RESULT_DIR
from tqdm import tqdm
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from LAC import LAC

lac = LAC()
embeddings = HuggingFaceEmbeddings(model_name="/home/zhangyh/models/bge-large-zh-v1.5")
fassi_kb = FAISS.load_local("/home/zhangyh/projs/llm-home/src/langchain_faiss/data/faiss/bge-chinese-1.5/cail_data",
                            embeddings=embeddings)


def get_dataset():
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as R:
        datasets = json.load(R)
    return datasets


def get_ans_postprocess(ans: str):
    tmp = list()
    for s in ans:
        if s in ["A", "B", "C", "D"]:
            tmp.append(s)
    return "".join(tmp)


def get_sematic(query: str):
    words_, labels_ = lac.run(query)
    type_lst = []
    for i, itm in enumerate(labels_):
        if itm == "nw":
            type_lst.append(words_[i])
    return type_lst


def get_relate_knowledge(query: str):
    result = fassi_kb.similarity_search_with_relevance_scores(query=query, k=3)
    return result


def get_dual_relate_kb(query: str):
    key = get_sematic(query)
    key_info = get_relate_knowledge(" ".join(key))
    sentence_info = get_relate_knowledge(query)
    return merge_duel_knowledge(key_info, sentence_info)


def merge_duel_knowledge(query_key: List[Tuple[Document, float]], query: List[Tuple[Document, float]]) -> List[str]:
    # TODO 写的有点拉，后面有时间再改吧！。
    tmp_lst = []
    for itm in query_key:
        if itm[0].page_content not in tmp_lst and len(itm[0].page_content) >= 20:
            tmp_lst.append(itm[0].page_content)

    for itm in query:
        if itm[0].page_content not in tmp_lst and len(itm[0].page_content) >= 20:
            tmp_lst.append(itm[0].page_content)

    return tmp_lst


def post_process_kb(doc: List[Tuple[Document, float]]):
    tmp_lst = list()
    for itm in doc:
        tmp_lst.append(itm[0].page_content)
    return tmp_lst


def get_acc(path="cail/Qwen1.5-7B-Chat_zeroshot_withrag.xlsx"):
    import pandas as pd
    df = pd.read_excel(RESULT_DIR + path)
    ans = df["ans"].to_list()
    output = df["output"].to_list()
    assert len(ans) == len(output)
    cnt = 0
    for i, itm in tqdm(enumerate(ans)):
        ans_ = "".join(eval(ans[i]))
        output_ = get_ans_postprocess(str(output[i]))
        if ans_ == output_:
            cnt += 1
    return float(cnt/len(ans))


if __name__ == "__main__":
    question = "根据我国《治安管理处罚条例》的有关规定，违反治安管理的，一般处以200元以下的罚款。并规定，受罚款处罚的人应当依法缴" \
               "纳罚款，无正当理由逾期不缴纳的，公安机关可以按日增加罚款1――5元。这里200元罚款和1――5元的罚款的性质是什么"

    c = get_relate_knowledge(question)
    print(len(c))

    all = merge_duel_knowledge(b, c)
    print(len(all))
    print(all)

