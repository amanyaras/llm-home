# class Config:
#     llm_model_name = '/home/chensm/LLM/models/Qwen-14B-Chat-Int8'  # 本地模型文件 or huggingface远程仓库
#     embedding_model_name = '/home/chensm/LLM/models/bge-large-zh-v1.5'  # 检索模型文件 or huggingface远程仓库
#     vector_store_path = '/home/chensm/LLM/FastChat-main/fastchat/serve/langchain_faiss/data/faiss/Flag'
#     vector_store_path2 = '/home/chensm/LLM/FastChat-main/fastchat/serve/langchain_faiss/data/faiss/Market'
#     vector_store_path3 = '/home/chensm/LLM/FastChat-main/fastchat/serve/langchain_faiss/data/faiss/Tmp'
#     docs_path = '/home/chensm/LLM/FastChat-main/fastchat/serve/langchain_faiss/data/已解密_副本机型支持模板2_去重.xlsx'

class Config:
    llm_model_name = '/home/zhangyh/models/Qwen-14B-Chat-Int8'  # 本地模型文件 or huggingface远程仓库
    embedding_model_name = '/home/zhangyh/models/bge-large-zh-v1.5'  # 检索模型文件 or huggingface远程仓库
    vector_store_path = '/home/zhangyh/projs/llm-home/src/langchain_faiss/data/faiss/wudao'
    # vector_store_path2 = '/home/zhangyh/projs/FastChat-main/fastchat/serve/langchain_faiss/data/faiss/Market'
    vector_store_path2 = '/home/zhangyh/projs/llm-home/src/langchain_faiss/data/faiss/Tmp'
    vector_store_path3 = '/home/zhangyh/projs/llm-home/src/langchain_faiss/data/faiss/Tmp'
    docs_path = '/home/zhangyh/projs/llm-home/src/langchain_faiss/data/已解密_副本机型支持模板2_去重.xlsx'

    vector_store_path_pinpai = '/home/zhangyh/projs/llm-home/src/langchain_faiss/data/faiss/pinpai'
    vector_store_path_xinghao = '/home/zhangyh/projs/llm-home/src/langchain_faiss/data/faiss/xinghao'
