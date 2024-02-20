"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"

from langchain_faiss.myapplication import MYApplication
from LAC import LAC


def get_logo_from_text(text: str, lac):
    words_, labels_ = lac.run(text.lower())
    logo_lst = []
    for i, itm in enumerate(labels_):
        if itm == "logo":
            logo_lst.append(words_[i])
    return logo_lst


# todo
def query_all_types_from_logo(logo: str, faiss):
    sv_lst = []
    # vector_store_path = "/home/zhangyh/FastChat-main/fastchat/serve/langchain_faiss/data/faiss/pinpai"
    # embeddings = HuggingFaceEmbeddings(model_name='/home/zhangyh/LLM/models/bge-large-zh-v1.5')
    # faiss = FAISS.load_local(vector_store_path, embeddings)
    result = faiss.similarity_search_with_relevance_scores(query=logo, k=50)
    for itm in result:
        page_content = itm[0].page_content
        if itm[1] > 0.9:
            sv_lst.append(page_content)
    return sv_lst


def get_type_from_text(text: str, lac):
    words_, labels_ = lac.run(text.lower())
    type_lst = []
    for i, itm in enumerate(labels_):
        if itm == "type":
            type_lst.append(words_[i])
    return type_lst


def difference(text: str, target: str):
    """
    text 表示原本的文本
    与小写的差异
    target 表示替换的字段
    """
    # todo
    idx = text.lower().index(target)
    return text[idx: len(target)+idx]


def query_all_types_from_types(type: str, faiss):
    sv_lst = []
    result = faiss.similarity_search_with_relevance_scores(query=type, k=50)
    for itm in result:
        page_content = itm[0].page_content
        if itm[1] > 0.68:
            sv_lst.append(page_content)
    return sv_lst


lac = LAC()
lac.load_customization('/home/zhangyh/projs/llm-home/src/langchain_faiss/data/json/types_.txt', sep="\n")
application = MYApplication()

ans = get_logo_from_text("苹果手机支持解锁吗?华为手机可不可以解锁？", lac)
print(ans)
