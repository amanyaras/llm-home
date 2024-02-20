from typing import List, Optional
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM,GenerationConfig
from langchain_faiss.config import Config
import torch

device = torch.device("cuda")
class LLMService(LLM):
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []
    tokenizer: object = None
    model: object = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "LLM"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        # response, _ = self.model.chat(
        #     self.tokenizer,
        #     prompt,
        #     history=self.history,
        #     # max_length=self.max_token,
        #     # temperature=self.temperature,
        # )
        # if stop is not None:
        #     response = enforce_stop_tokens(response, stop)
        # self.history = self.history + [[None, response]]

        # baichuan
        # message = []
        # message.append({"role": "user", "content": prompt})
        # response = self.model.chat(self.tokenizer, message)

        # ziya-13B
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        generate_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=4000,
                    do_sample = True,
                    top_p = 0.8,
                    temperature = 0.1,
                    repetition_penalty=1.,
                    eos_token_id=self.tokenizer.encode("</s>"),
                    )
        response = self.tokenizer.batch_decode(generate_ids)[0]

        return response

    def load_model(self, model_name_or_path: str = "ClueAI/ChatYuan-large-v2"):
        """
        加载大模型LLM
        :return:
        """
        print(Config.llm_model_name)
        # baichuan
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     Config.llm_model_name, use_fast=False, trust_remote_code=True
        # )
        # self.model = AutoModelForCausalLM.from_pretrained(Config.llm_model_name,torch_dtype=torch.float16, trust_remote_code=True).quantize(8).cuda()
        # self.model.generation_config = GenerationConfig.from_pretrained(Config.llm_model_name) #baichuan
        # self.model = self.model.eval()

        # qwen
        self.tokenizer = AutoTokenizer.from_pretrained(Config.llm_model_name, trust_remote_code=True, revision = 'v1.0.0')
        self.model = AutoModelForCausalLM.from_pretrained(Config.llm_model_name,device_map = "auto",trust_remote_code=True)
        self.model = self.model.eval()

        # chatglm2
        # self.tokenizer = AutoTokenizer.from_pretrained(Config.llm_model_name, trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(Config.llm_model_name,device_map = "auto",trust_remote_code=True).quantize(8)
        # self.model = self.model.eval()

        # chatglm3
        # self.tokenizer = AutoTokenizer.from_pretrained(Config.llm_model_name, trust_remote_code=True)
        # self.model = AutoModel.from_pretrained(Config.llm_model_name, trust_remote_code=True).half().cuda()
        # self.model = self.model.eval()

        # ziya-13B
        # model_dir = snapshot_download('Fengshenbang/Ziya-Reader-13B-v1.0', revision = 'v1.0.0')
        
        # self.model = AutoModelForCausalLM.from_pretrained(Config.llm_model_name,device_map='auto', load_in_8bit=True)
        # self.model = AutoModelForCausalLM.from_pretrained(Config.llm_model_name,device_map='auto')
        #
        # self.tokenizer = AutoTokenizer.from_pretrained(Config.llm_model_name, use_fast=False)
        # self.model = self.model.eval()


if __name__ == '__main__':
    chatLLM = LLMService()
    chatLLM.load_model()
    a = chatLLM._call("123")
    print(a)