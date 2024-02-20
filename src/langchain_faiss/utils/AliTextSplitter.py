from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List
# from modelscope.pipelines import pipeline

class AliTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    # 此处采取的文档语义分割模型为达摩院开源的nlp_bert_document-segmentation_chinese-base，论文见https://arxiv.org/abs/2107.09278
    def split_text(self, text: str) -> List[str]:
        

        lines = text.split('\n\n\n')
        sent_list = [l.strip() for l in lines]

        print('*********************', len(sent_list ))

        return sent_list


if __name__ == '__main__':
    # 使用text_splitter对文档进行分割
    text_splitter = AliTextSplitter()