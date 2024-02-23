import os
import sys
sys.path.insert(0, "/home/zhangyh/projs/llm-home/src")
from builtins import print
from time import sleep
# from langchain.document_loaders import UnstructuredFileLoader, TextLoader, DirectoryLoader, JSONLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain_faiss.config import Config
from langchain_faiss.utils.AliTextSplitter import AliTextSplitter
import json

import pandas as pd
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class DocumentService(object):
    def __init__(self):

        self.config = Config.vector_store_path
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.embedding_model_name)
        self.docs_path = Config.docs_path
        self.vector_store_path = Config.vector_store_path
        self.vector_store = None

        self.vector_store_path2 = Config.vector_store_path2
        self.market_vector_store = None

        self.vector_store_path_tmp = Config.vector_store_path3
        self.tmp_vector_store = None

        self.vector_store_path_pinpai = Config.vector_store_path_pinpai
        self.vector_store_path_xinghao = Config.vector_store_path_xinghao
        self.pinpai_vector_store = None

    def init_source_vector(self):
        """
        初始化本地知识库向量
        :return:
        """
        datas = ""
        df = pd.read_excel(self.docs_path)

        for index, row in df.iterrows():
            if str(row['品牌']) == "华为" and "Mate30" in str(row['型号']):
                data = "品牌：" + str(row['品牌']) + "\n" + "型号：" + str(row['型号']) + "\n" + "是否支持解锁：" + str(row['支持']) + "\n" + "备注：" + "鸿蒙3.0不支持," + str(row['备注'])
                datas = datas + data + "\n\n\n"
                continue
            data = "品牌：" + str(row['品牌']) + "\n" + "型号：" + str(row['型号']) + "\n" + "是否支持解锁：" + str(row['支持']) + "\n" + "备注：" + str(row['备注'])
            datas = datas + data + "\n\n\n"


        splitter = AliTextSplitter()
        split_text = splitter.split_text(datas)
        print(type(split_text))
        # 采用embeding模型对文本进行向量化
        
        self.vector_store = FAISS.from_texts(split_text, self.embeddings)
        # 把结果存到faiss索引里面
        self.vector_store.save_local(self.vector_store_path)

    def load_vector_store(self):
        self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings)
        return self.vector_store

    def load_market_vector_store(self):
        self.market_vector_store = FAISS.load_local(self.vector_store_path2, self.embeddings)
        return self.market_vector_store

    def load_tmp_vector_store(self):
        self.tmp_vector_store = FAISS.load_local(self.vector_store_path_tmp, self.embeddings)
        return self.tmp_vector_store

    def load_pinpai_vector_store(self):
        self.pinpai_vector_store = FAISS.load_local(self.vector_store_path_pinpai, self.embeddings)
        return self.pinpai_vector_store


    def load_xinghao_vector_store(self):
        self.xinghao_vector_store = FAISS.load_local(self.vector_store_path_xinghao, self.embeddings)
        return self.xinghao_vector_store

    def encode_keyword_only(self,
                            data: List[Dict],
                            key_list: List[str],
                            value_list: List[str]) -> List:
        embedding_list = list()
        answer_list = list()
        tmp = ""
        for i, itm in enumerate(data):
            tmp = ""
            for key in key_list:
                tmp += str(itm[key])
            # tmp += " "
            embedding_list.append(tmp.lower())
        for i, itm in enumerate(data):
            tmp = ''
            for value in value_list:
                tmp += str(itm[value])
                tmp += " "
            answer_list.append(tmp)

        embedding_model = SentenceTransformer(Config.embedding_model_name)
        embeddings = embedding_model.encode(embedding_list)

        assert len(embedding_list) == len(answer_list)
        va_lst = list()
        for fin in range(len(data)):
            va_lst.append((answer_list[fin], embeddings[fin]))
        return va_lst


if __name__ == '__main__':
    s = DocumentService()
    all_data = []
    for root, dirs, files in os.walk("/home/zhangyh/projs/llm-home/cail_data/reference_book/"):
        for file in files:
            if file.endswith('.txt'):
                # 打印或处理找到的txt文件
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as R:
                    content = R.read()
                    tmp_lst = eval(content)
                    all_data += tmp_lst
    print(len(all_data))

    fassi = FAISS.from_texts(all_data, s.embeddings)
    fassi.save_local("/home/zhangyh/projs/llm-home/src/langchain_faiss/data/faiss/bge-chinese-1.5")


    # fassi = FAISS.load_local(Config.vector_store_path, embeddings=s.embeddings)
    #
    result = fassi.similarity_search_with_relevance_scores(query="Mate9 Pro(LON-TL00)".lower(), k=10)

    """
    data = pd.read_excel("/home/zhangyh/FastChat-main/fastchat/serve/langchain_faiss/data/已解密_副本机型支持模板2_去重.xlsx") \
    .to_numpy().tolist()
    with open("/home/zhangyh/FastChat-main/fastchat/serve/langchain_faiss/data/json/types_new.txt", 'r') as R:
        data_ = R.readlines()
    sv_lst = list()
    for itm in range(len(data)):
        sv_lst.append({
            "品牌": data[itm][0],
            "型号": data[itm][1],
            "支持": str(data[itm][0]) + str(data[itm][1]) + "支持",
            "问题": str(data[itm][0]) + str(data[itm][1]) + "\n" + "是否支持解锁：" + str(data[itm][2]) + "\n" + "备注：" + str(data[itm][6]),
            "回答": str(data[itm][0]) + str(data[itm][1]) + "是否支持解锁：" + str(data[itm][2]) + "\n" + "备注：如果是旧机型，可能因更新不一定成功"
        })
    sv_lst = []
    for itm in data_:
        sv_lst.append({
            "品牌": itm.split("/logo")[0],
            "型号": itm.split("\t")[1].replace("/type", "")
        })

    wuytu = s.encode_keyword_only(sv_lst, key_list=["型号"], value_list=["品牌", "型号"])
    fassi = FAISS.from_embeddings(wuytu, s.embeddings)
    fassi.save_local(Config.vector_store_path_xinghao)
    # print(wuytu)
    # 0.62 0.39
    # fassi = FAISS.load_local(Config.vector_store_path, embeddings=s.embeddings)
    #
    result = fassi.similarity_search_with_relevance_scores(query="Mate9 Pro(LON-TL00)".lower(), k=10)
    print(result)
    #
    # import paddlehub as hub
    #
    # lac = hub.Module(name="lac")
    # test_text = [result[0][0].page_content]
    #
    # results = lac.cut(text=test_text, use_gpu=False, batch_size=1, return_tag=True)
    # print(results)
    #
    # test_text = ["支不支持解锁荣耀的Play5(HJC-AN90)"]
    # results = lac.cut(text=test_text, use_gpu=False, batch_size=1, return_tag=True)
    # print(results)
    ###将文本分块向量化存储起来
    # s.init_source_vector()
    """
