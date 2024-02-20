"""案件名称、、审理程序、案由、法律依据、全文"""
# import json
from langchain.vectorstores.faiss import FAISS
# from fastchat.serve.langchain_faiss.config import Config
# sv_lst = []
# for i in range(100):
#     sv_lst.append({
#         "案件名称": i,
#         "案件类型":i,
#         "案由":i,
#         "法律依据":i,
#         "全文":i
#     })
# with open("tmp.json", 'w', encoding='utf-8') as W:
#     json.dump(sv_lst, W, indent=4, ensure_ascii=False)
#
# from langchain.vectorstores import FAISS
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
#
#
# qus = ["\nqus" + 'a'*100 + str(i) for i in range(0, 10)]
# ans = ["\nans" + 'b'*100 + str(j) for j in range(0, 10)]
# embeddings = HuggingFaceEmbeddings(model_name=Config.embedding_model_name)
# text_embeddings = embeddings.embed_documents(qus)
# text_embedding_pairs = zip(ans, text_embeddings)
# faiss = FAISS.from_embeddings(text_embedding_pairs, embeddings)
import json
import pandas as pd

# lst = []
# data = pd.read_excel("/home/zhangyh/FastChat-main/fastchat/serve/langchain_faiss/data/已解密_副本机型支持模板2_去重.xlsx")
# kinds = data["型号"].to_numpy().tolist()
# for i, itm in enumerate(kinds):
#     if itm not in lst:
#
#         itm = str(itm)
#         lst.append(itm)
# print(lst)
# with open("./kinds_xingaho.json", 'w', encoding='utf-8') as W:
#     # data = json.load(W)
# # print(data)
#     json.dump(lst, W, ensure_ascii=False, indent=4)
# from LAC import LAC
#
final_relat_data = '回答:华为P30(ELE-TL00)是否支持解锁：支持	备注：如果是旧机型，可能因更新不一定成功'
text = "华为mate40支持解锁吗"
from LAC import LAC
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
lac = LAC()
lac.load_customization('./types.txt', sep="\n")


def get_logo_from_text(text: str):
    words_, labels_ = lac.run(text)
    return words_[labels_.index("logo")] if "logo" in labels_ else None


def get_type_from_text(text: str):
    words_, labels_ = lac.run(text)
    return words_[labels_.index("type")] if "type" in labels_ else None


# todo
def query_all_types_from_logo(logo: str):
    sv_lst = []
    vector_store_path = "/home/zhangyh/FastChat-main/fastchat/serve/langchain_faiss/data/faiss/pinpai"
    embeddings = HuggingFaceEmbeddings(model_name='/home/zhangyh/LLM/models/bge-large-zh-v1.5')
    faiss = FAISS.load_local(vector_store_path, embeddings)
    result = faiss.similarity_search_with_relevance_scores(query=logo, k=10)
    for itm in result:
        page_content = itm[0].page_content
        if itm[1] > 0.9:
            sv_lst.append(page_content)
    return sv_lst





def match_(text, final_relat_data):
    lac_text = lac.run(text)
    lac_final_relat_data = lac.run(final_relat_data)
    text_logo_lst = []
    text_type_lst = []
    final_relat_data_logo_lst = []
    final_relat_data_type_lst = []
    # logo type
    for i, itm in enumerate(lac_final_relat_data[1]):
        if itm == "logo":
            final_relat_data_logo_lst.append(lac_final_relat_data[0][i])
        if itm == "type":
            final_relat_data_type_lst.append(lac_final_relat_data[0][i])

    type = "".join(final_relat_data_type_lst)

    for j, itm in enumerate(lac_text[1]):
        if itm == "type":
            text_type_lst.append(lac_text[0][j])
        if itm == "logo":
            text_logo_lst.append(lac_text[0][j])

    for l in text_logo_lst:
        if l.lower().replace(" ", "") != final_relat_data_logo_lst[0].lower().replace(" ", ""):
            final_relat_data = "该机型未找到相关记录，请联系4008886688客服确认"

    for t in final_relat_data_type_lst:
        if t.lower().replace(" ", "") not in type.lower().replace(" ", ""):
            final_relat_data = "该机型未找到相关记录，请联系4008886688客服确认"

    return final_relat_data


def multify_n(cnt):
    cnt = str(cnt)
    tmp_lst = []
    tmp_lst.append(cnt)
    tmp_lst.append(cnt.lower())
    if "(" in cnt and ")" in cnt:
        tmp_cnt = cnt.split("(")[0]
        # tmp_lst.append(str(tmp_cnt))
        tmp_lst.append(str(tmp_cnt.lower()))
    # if " " in cnt:
    #     tmp_cnt = cnt.split(" ")[0]
    #     # tmp_lst.append(str(tmp_cnt))
    #     tmp_lst.append(str(tmp_cnt.lower()))
    return tmp_lst


print(get_logo_from_text(text))
print(query_all_types_from_logo("苹果"))
f = match_(text, final_relat_data)
print(f)
# import pandas as pd
#
types = pd.read_excel("../已解密_副本机型支持模板2_去重.xlsx")
xinghao = types["型号"].to_numpy().tolist()
logo = types["品牌"].to_numpy().tolist()
print(len(xinghao))
sv_lst = []
# for itm in xinghao:
#     itm_lst = multify_n(itm)
#     for tmp in itm_lst:
#         sv_lst.append(str(tmp).lower() + r"/type" + "\n")
# for itm in logo:
#     if str(itm).lower() not in sv_lst:
#         sv_lst.append(str(itm).lower() + r"/logo" + "\n")
#     # sv_lst.append(str(itm) + r"/logo" + "\n")

same_dict = {
    "三星": "sumsung",
    "华为": "huawei",
    "魅族": "meizu",
    "荣耀": "honor",
    "小米": "xiaomi",
    "中兴": "zte",
    "联想": "lenovo",
}

for idx in range(len(types)):

    itm_lst = multify_n(xinghao[idx])
    for tmp in itm_lst:
        sv_lst.append(str(logo[idx]).lower() + r"/logo" + "\t" + str(tmp).lower() + r"/type" + "\n")
        if str(logo[idx]) in same_dict:
            sv_lst.append(same_dict[str(logo[idx]).lower()] + r"/logo" + "\t" + str(tmp).lower() + r"/type" + "\n")

"""
三星 sumsung
华为 huawei
魅族 meizu
荣耀 honor
小米 xiaomi
中兴 zte
联想 lenovo
"""
# sv_lst.append("sumsung" + r"/logo" + "\n")
# sv_lst.append("huawei" + r"/logo" + "\n")
# sv_lst.append("meizu" + r"/logo" + "\n")
# sv_lst.append("honor" + r"/logo" + "\n")
# sv_lst.append("xiaomi" + r"/logo" + "\n")
# sv_lst.append("zte" + r"/logo" + "\n")
# sv_lst.append("Lenovo" + r"/logo" + "\n")




"""
ef get_ans(cnt,
url="http://192.168.1.15:23117/worker_ge
prev =0 ans = [] params = {
"model": "sft_30k_zh_v4"，
"prompt":"这是由厦门市美亚柏科信息股份有限公司研发
" USER: {问题}Assistant:",
"temperature": 0.7,
"max_new_tokens" :4096,
"stop_token_ids": None,
"echo": False}

params["prompt"] = params["prompt"].format(cnt) response = requests.post(

"""


final_lst = []
for itm in sv_lst:
    if itm not in final_lst:
        final_lst.append(itm)

with open("./types_new.txt", 'w') as W:
    W.writelines(final_lst)



# nz = results[0]['word'][results[0]['tag'].index('nz')]
# nz_lst = [itm for itm in nz]
# print(nz_lst)
# for result in results:
#     print(result['word'])
#     print(result['tag'])


# 批量样本输入, 输入为多个句子组成的list，平均速率更快
# texts = [u"LAC是个优秀的分词工具", u"百度是一家高科技公司"]
# lac_result = lac.run(texts)
# from LAC import LAC
#
# # 装载LAC模型
# lac = LAC(mode='lac')
#
# # 单个样本输入，输入为Unicode编码的字符串
# text = u"LAC是个优秀的分词工具"
# lac_result = lac.run(text)
#
# # 批量样本输入, 输入为多个句子组成的list，平均速率更快
# texts = [u"LAC是个优秀的分词工具", u"百度是一家高科技公司"]
# lac_result = lac.run(texts)