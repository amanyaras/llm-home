import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from langchain_faiss.config import Config
from langchain_faiss.document import DocumentService
from langchain_faiss.llm import LLMService
# from text2vec import Similarity
import torch


class MYApplication(object):

    def __init__(self):
        self.config = Config
        self.llm_service = LLMService()
        ###加载llm和知识库向量
        # print("load llm model ")
        self.llm_service.load_model(model_name_or_path=self.config.llm_model_name)
        self.doc_service = DocumentService()
        print("load documents")
        # self.doc_service.load_vector_store()
        self.vector_store = self.doc_service.load_vector_store()

        self.doc_service_market = DocumentService()
        print("load market documents")
        self.market_vector_store = self.doc_service_market.load_market_vector_store()

        self.doc_service_tmp = DocumentService()
        print("load tmp documents")
        self.tmp_vector_store = self.doc_service_tmp.load_tmp_vector_store()
        # self.sim_service = Similarity(model_name_or_path=self.config.embedding_model_name)
        self.doc_service_pinpai = DocumentService()
        self.vector_store_pinpai = self.doc_service_pinpai.load_pinpai_vector_store()

        self.doc_service_xinghao = DocumentService()
        self.vector_store_xinghao = self.doc_service_xinghao.load_xinghao_vector_store()

    def get_final_related_data(self, relat_data):
        final_relat_data = ""
        if len(relat_data) == 1 and "备注：如是旧机型，可能因更新不一定成功" in relat_data[0][0].page_content:
            if relat_data[0][1] > 0.8:
                return ""
            return relat_data[0][0].page_content.replace('\n', '\t') + "\n"
        for document in relat_data:
            data, score = document  # data为检索内容，score为相似检索的分数
            page_content = data.page_content
            if len(final_relat_data) < 1000 and score <= 0.9:  # 这里的判断语句是，防止检索内容大于1000
                final_relat_data = final_relat_data + page_content.replace('\n', '\t') + "\n"
        return final_relat_data


    def search_from_vectorstore(self, query, vector, search_type, k=50):
        # 查看检索内容，并计算与输入的关联程度
        q = "为这个句子生成表示以用于检索相关文章："
        if search_type == 'similarity_search':
            # relat_data = self.doc_service.vector_store.similarity_search(q + query,k=1,fetch_k=4)
            relat_data = vector.similarity_search(q + query, k=k, fetch_k=4)
            final_relat_data = self.get_final_related_data(relat_data)

        if search_type == 'max_marginal_relevance_search':
            # relat_data = self.doc_service.vector_store.max_marginal_relevance_search(q + query,k=3,fetch_k=4)
            relat_data = vector.max_marginal_relevance_search(q + query, k=k, fetch_k=4)
            final_relat_data = self.get_final_related_data(relat_data)

        if search_type == 'similarity_search_with_score':
            # relat_data = self.doc_service.vector_store.similarity_search_with_score(q + query,k=k)
            relat_data = vector.similarity_search_with_score(q + query, k=k)
            # print(type(relat_data))
            # print(relat_data)
            final_relat_data = self.get_final_related_data(relat_data)


        # print("**************************************")
        # print("\n")
        # 问题：苹果手机如何解锁？\n答案：需要读取具体版本号确认是否支持解锁，通过公众号关注美亚柏科服务之星，或者拨打4008886688热线联系在线技术咨询解锁支持。
        # print("检索知识库结果：")
        # print(final_relat_data)
        # print("**************************************")
        # print("检索信息与输入的相似分数：")
        # print(score)
        # print("**************************************")
        return final_relat_data
        

    def get_llm_answer(self, query=''):
        # prompt_template = """请回答下列问题:
        #                     {}""".format(query)

        # ziya-13B
        prompt_template = """<human>: 
                            {}\n<bot>:""".format(query)
        print(prompt_template)
        ### 基于大模型的问答
        result = self.llm_service._call(prompt_template)
        return result

    def get_knowledge_based_answer(self, query,
                                   history_len=5,
                                   temperature=0.1,
                                   top_p=0.9,
                                   top_k=4,
                                   chat_history=[]):
        
        self.llm_service.history = chat_history[-history_len:] if history_len > 0 else []

        self.llm_service.temperature = temperature
        self.llm_service.top_p = top_p

        # 将检索的知识库内容与query结合，传入模型
        final_relat_data = self.search_from_vectorstore(query, search_type='similarity_search_with_score',k=50)
        # 声明一个知识库问答llm,传入之前初始化好的llm和向量知识搜索服务
        if final_relat_data != []:
            # prompt_template = """基于以下已知信息，详细和专业的来回答用户的问题。
            #                                 如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
            #                                 已知内容（该内容是手机取证支持的品牌与相应的型号）:
            #                                 {}
            #                                 详细回答下面问题:
            #                                 {}，这个答案必须根据提供的信息回答，如果提供的信息和问题不一致，不要根据提供的信息回答，并且不一致的情况不能胡编乱造，没有就回答我不知道。""".format(final_relat_data,query)

            # ziya-13B
            # final_relat_data = final_relat_data.strip().split('\n')
            # for i in range(0,len(final_relat_data)):
            #     I = i + 1
            #     final_relat_data[i] = '[' + str(I) + ']' + final_relat_data[i] + '<eod>\n'

            prompt_template = """<human>: 给定问题：{}\n检索结果：{}请阅读理解上面多个检索结果，正确地回答问题。只能根据相关的检索结果或者知识回答，禁止编造；如果没有相关结果，请回答“都不相关，我不知道”。\n<bot>:""".format(query,final_relat_data)
            # print(prompt_template)
            result = self.llm_service._call(prompt_template)
            result = {"question": query,
                    "result": result,
                    "Document": final_relat_data}
            return result
        else:
            prompt_template = """请回答下列问题:
                            {}""".format(query)
            ### 基于大模型的问答
            result = self.llm_service._call(prompt_template)
            return result


    def get_prompt_based_knowledge(self, query, final_relat_data):
        # 修改Prompt
        # prompt_template = """基于以下已知信息，详细和专业的来回答用户的问题。
        #                                 如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
        #                                 已知内容（该内容是手机取证支持的品牌与相应的型号）:
        #                                 {}
        #                                 详细回答下面问题:
        #                                 {}。这个答案必须根据提供的信息回答，如果提供的信息和问题不一致，不要根据提供的信息回答，并且不一致的情况不能胡编乱造，没有就回答我不知道。""".format(
        #     final_relat_data, query)
        # prompt_template = """基于以下已知信息，详细和专业的来回答用户的问题。已知内容（该内容是手机取证支持的品牌与相应的型号）:{}。回答下面问题:{}。只能根据已知内容进行作答，禁止编造内容，如果没有相关结果，请回答"根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。""".format(
        #     final_relat_data, query)

        prompt_template = """已知信息：{}\n上述信息字母不区分大小写。根据上述已知信息，简洁和专业的来回答用户的问题。问题是：{}。\n如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。
        """.format(
            final_relat_data, query)

        # ziya-13B
        # final_relat_data = final_relat_data.strip().split('\n')
        # for i in range(0, len(final_relat_data)):
        #     I = i + 1
        #     final_relat_data[i] = '[' + str(I) + ']' + final_relat_data[i] + '<eod>\n'

        # prompt_template = """<human>: 给定问题：{}\n检索结果：{}请阅读理解上面多个检索结果，正确地回答问题。只能根据相关的检索结果或者知识回答，禁止编造；如果没有相关结果，请回答“都不相关，我不知道”。\n<bot>:""".format(
        #     query, final_relat_data)
        # print(prompt_template)
        return prompt_template


    def get_prompt_based_knowledge_unlock(self, query, final_relat_data):
        # prompt_template = """已知信息：{}\n注意：1、()内的内容不同也算是不同的型号。2、上述信息字母不区分大小写和空格。3、已知信息不一是全的。4、你不能输出相同的型号。根据上述已知的可以解锁的手机型号来回答问题。问题是：{}\n如果无法从中得到答案，请说 “该机型未找到相关记录，请联系4008886688客服确认”, 不允许在答案中添加编造成分，答案请使用中文。
        # """.format(final_relat_data.replace("\n \n", "\n"), query)
        # return prompt_template
        instruct_info = "作为美亚柏科产品售后支持助手，请专业、详尽的回答用户提问。"
        tele_template = "{instruct_info}\n" \
                        "注意：你需要遵循以下规则：\n" \
                        "1、()内的内容不同也算是不同的型号，只符合一部分也算是不同型号。\n" \
                        "2、上述信息字母不区分大小写和空格。\n" \
                        "3、已知信息中的每一条信息通过\t进行分隔。\n" \
                        "4、你的回答不能重复，你不能输出相同的型号。\n" \
                        "5、不允许在答案中添加编造成分，答案需要使用中文。\n" \
                        "在符合上述规则的前提下你需要做的是：判断问题中所询问的信息是否出现在已知信息中。\n" \
                        "如果问题中的信息没有出现在已知信息中则回答：\n" \
                        "“该机型未找到相关记录。微信关注美亚柏科服务之星，联系人工客服咨询，或拨打客服电话400-888-6688。”\n" \
                        "已知信息：{index_info}。问题是：{query}"\
                        .format(instruct_info=instruct_info, index_info=final_relat_data, query=query)
        return tele_template


if __name__ == '__main__':
    device = torch.device("cuda")
    application = MYApplication()
    while 1:
        print("输入问题：")
        query = input()
        # query = '华为手机P30 Pro是否支持取证'
        print("大模型自己回答的结果")
        result = application.get_llm_answer(query)
        print(result)
        print("大模型+知识库后回答的结果")
        result = application.get_knowledge_based_answer(query)
        final_relat_data = ""
        application.get_prompt_based_knowledge(query, final_relat_data)
        print(result)
    
    