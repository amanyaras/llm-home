"""
The gradio demo server for chatting with a single model.
"""

import argparse
from collections import defaultdict
import datetime
import json
import os
import random
import time
import uuid

import gradio as gr
import requests

from fastchat.conversation import SeparatorStyle
from fastchat.constants import (
    LOGDIR,
    WORKER_API_TIMEOUT,
    ErrorCode,
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    RATE_LIMIT_MSG,
    SERVER_ERROR_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SESSION_EXPIRATION_TIME,
)
from fastchat.model.model_adapter import (
    get_conversation_template,
    ANTHROPIC_MODEL_LIST,
)
from fastchat.model.model_registry import get_model_info, model_info
from fastchat.serve.api_provider import (
    anthropic_api_stream_iter,
    openai_api_stream_iter,
    palm_api_stream_iter,
    gemini_api_stream_iter,
    bard_api_stream_iter,
    mistral_api_stream_iter,
    nvidia_api_stream_iter,
    ai2_api_stream_iter,
    init_palm_chat,
)
from fastchat.utils import (
    build_logger,
    moderation_filter,
    get_window_url_params_js,
    get_window_url_params_with_tos_js,
    parse_gradio_auth_creds,
)


logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "FastChat Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True, visible=True)
disable_btn = gr.Button(interactive=False)
invisible_btn = gr.Button(interactive=False, visible=False)

controller_url = None
enable_moderation = False

acknowledgment_md = """
### Terms of Service

Users are required to agree to the following terms before using the service:

The service is a research preview. It only provides limited safety measures and may generate offensive content.
It must not be used for any illegal, harmful, violent, racist, or sexual purposes.
The service collects user dialogue data and reserves the right to distribute it under a Creative Commons Attribution (CC-BY) or a similar license.
Additionally, Bard is offered on LMSys for research purposes only. To access the Bard product, please visit its [website](http://bard.google.com).

### Acknowledgment
We thank [Kaggle](https://www.kaggle.com/), [MBZUAI](https://mbzuai.ac.ae/), [a16z](https://www.a16z.com/), [Together AI](https://www.together.ai/), [Anyscale](https://www.anyscale.com/), [HuggingFace](https://huggingface.co/) for their generous [sponsorship](https://lmsys.org/donations/).

<div class="image-about">
    <img src="https://storage.googleapis.com/public-arena-asset/kaggle.png" alt="Kaggle">
    <img src="https://storage.googleapis.com/public-arena-asset/mbzuai.jpeg" alt="MBZUAI">
    <img src="https://storage.googleapis.com/public-arena-asset/a16z.jpeg" alt="a16z">
    <img src="https://storage.googleapis.com/public-arena-asset/together.png" alt="Together AI">
    <img src="https://storage.googleapis.com/public-arena-asset/anyscale.png" alt="AnyScale">
    <img src="https://storage.googleapis.com/public-arena-asset/huggingface.png" alt="HuggingFace">
</div>
"""

ip_expiration_dict = defaultdict(lambda: 0)

# JSON file format of API-based models:
# {
#     "vicuna-7b": {
#         "model_name": "vicuna-7b-v1.5",
#         "api_base": "http://8.8.8.55:5555/v1",
#         "api_key": "password",
#         "api_type": "openai", # openai, anthropic, palm, mistral
#         "anony_only": false,  # whether to show this model in anonymous mode only
#     },
# }
api_endpoint_info = {}


class State:
    def __init__(self, model_name):
        self.conv = get_conversation_template(model_name)
        self.conv_id = uuid.uuid4().hex
        self.skip_next = False
        self.model_name = model_name

        if model_name in ["palm-2", "gemini-pro"]:
            self.palm_chat = init_palm_chat(model_name)

    def to_gradio_chatbot(self):
        return self.conv.to_gradio_chatbot()

    def dict(self):
        base = self.conv.dict()
        base.update(
            {
                "conv_id": self.conv_id,
                "model_name": self.model_name,
            }
        )
        return base


def set_global_vars(controller_url_, enable_moderation_):
    global controller_url, enable_moderation
    controller_url = controller_url_
    enable_moderation = enable_moderation_


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list(controller_url, register_api_endpoint_file):
    global api_endpoint_info
    if controller_url:
        ret = requests.post(controller_url + "/refresh_all_workers")
        assert ret.status_code == 200
        ret = requests.post(controller_url + "/list_models")
        models = ret.json()["models"]
    else:
        models = []

    # Add API providers
    if register_api_endpoint_file:
        api_endpoint_info = json.load(open(register_api_endpoint_file))
        models += list(api_endpoint_info.keys())

    models = list(set(models))
    visible_models = models.copy()
    for mdl in visible_models:
        if mdl not in api_endpoint_info:
            continue
        mdl_dict = api_endpoint_info[mdl]
        if mdl_dict["anony_only"]:
            visible_models.remove(mdl)

    priority = {k: f"___{i:03d}" for i, k in enumerate(model_info)}
    models.sort(key=lambda x: priority.get(x, x))
    visible_models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"All models: {models}")
    logger.info(f"Visible models: {visible_models}")
    return visible_models, models


def load_demo_single(models, url_params):
    selected_model = models[0] if len(models) > 0 else ""
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            selected_model = model

    dropdown_update = gr.Dropdown(choices=models, value=selected_model, visible=True)

    state = None
    return state, dropdown_update


def load_demo(url_params, request: gr.Request):
    global models

    ip = get_ip(request)
    logger.info(f"load_demo. ip: {ip}. params: {url_params}")
    ip_expiration_dict[ip] = time.time() + SESSION_EXPIRATION_TIME

    if args.model_list_mode == "reload":
        models, all_models = get_model_list(
            controller_url,
            args.register_api_endpoint_file,
        )

    return load_demo_single(models, url_params)


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"upvote. ip: {ip}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"downvote. ip: {ip}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"flag. ip: {ip}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"regenerate. ip: {ip}")
    state.conv.update_last_message(None)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def clear_history(request: gr.Request):
    ip = get_ip(request)
    logger.info(f"clear_history. ip: {ip}")
    state = None
    return (state, [], "") + (disable_btn,) * 5


def get_ip(request: gr.Request):
    if "cf-connecting-ip" in request.headers:
        ip = request.headers["cf-connecting-ip"]
    else:
        ip = request.client.host
    return ip


def add_text(state, model_selector, text, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"add_text. ip: {ip}. len: {len(text)}")

    if state is None:
        state = State(model_selector)

    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "") + (no_change_btn,) * 5

    flagged = moderation_filter(text, [state.model_name])
    if flagged:
        logger.info(f"violate moderation. ip: {ip}. text: {text}")
        # overwrite the original text
        text = MODERATION_MSG

    conv = state.conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), CONVERSATION_LIMIT_MSG) + (
            no_change_btn,
        ) * 5

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


def model_worker_stream_iter(
    conv,
    model_name,
    worker_addr,
    prompt,
    temperature,
    repetition_penalty,
    top_p,
    max_new_tokens,
):
    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "stop": conv.stop_str,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }
    logger.info(f"==== request ====\n{gen_params}")

    # Stream output
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json=gen_params,
        stream=True,
        timeout=WORKER_API_TIMEOUT,
    )
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            yield data


def is_limit_reached(model_name, ip):
    monitor_url = "http://localhost:9090"
    try:
        ret = requests.get(
            f"{monitor_url}/is_limit_reached?model={model_name}&user_id={ip}", timeout=1
        )
        obj = ret.json()
        return obj
    except Exception as e:
        logger.info(f"monitor error: {e}")
        return None


def bot_response(
    state,
    temperature,
    top_p,
    max_new_tokens,
    request: gr.Request,
    apply_rate_limit=True,
):
    ip = get_ip(request)
    logger.info(f"bot_response. ip: {ip}")
    start_tstamp = time.time()
    temperature = float(temperature)
    top_p = float(top_p)
    max_new_tokens = int(max_new_tokens)

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        state.skip_next = False
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if apply_rate_limit:
        ret = is_limit_reached(state.model_name, ip)
        if ret is not None and ret["is_limit_reached"]:
            error_msg = RATE_LIMIT_MSG + "\n\n" + ret["reason"]
            logger.info(f"rate limit reached. ip: {ip}. error_msg: {ret['reason']}")
            state.conv.update_last_message(error_msg)
            yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
            return

    conv, model_name = state.conv, state.model_name
    model_api_dict = (
        api_endpoint_info[model_name] if model_name in api_endpoint_info else None
    )

    if model_api_dict is None:
        # Query worker address
        ret = requests.post(
            controller_url + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

        # No available worker
        if worker_addr == "":
            conv.update_last_message(SERVER_ERROR_MSG)
            yield (
                state,
                state.to_gradio_chatbot(),
                disable_btn,
                disable_btn,
                disable_btn,
                enable_btn,
                enable_btn,
            )
            return

        # Construct prompt.
        # We need to call it here, so it will not be affected by "▌".
        prompt = conv.get_prompt()

        # Set repetition_penalty
        if "t5" in model_name:
            repetition_penalty = 1.2
        else:
            repetition_penalty = 1.0

        stream_iter = model_worker_stream_iter(
            conv,
            model_name,
            worker_addr,
            prompt,
            temperature,
            repetition_penalty,
            top_p,
            max_new_tokens,
        )
    elif model_api_dict["api_type"] == "openai":
        prompt = conv.to_openai_api_messages()
        stream_iter = openai_api_stream_iter(
            model_api_dict["model_name"],
            prompt,
            temperature,
            top_p,
            max_new_tokens,
            api_base=model_api_dict["api_base"],
            api_key=model_api_dict["api_key"],
        )
    elif model_api_dict["api_type"] == "anthropic":
        prompt = conv.get_prompt()
        stream_iter = anthropic_api_stream_iter(
            model_name, prompt, temperature, top_p, max_new_tokens
        )
    elif model_api_dict["api_type"] == "palm":
        stream_iter = palm_api_stream_iter(
            model_name,
            state.palm_chat,
            conv.messages[-2][1],
            temperature,
            top_p,
            max_new_tokens,
        )
    elif model_api_dict["api_type"] == "gemini":
        stream_iter = gemini_api_stream_iter(
            model_api_dict["model_name"],
            conv,
            temperature,
            top_p,
            max_new_tokens,
            api_key=model_api_dict["api_key"],
        )
    elif model_api_dict["api_type"] == "bard":
        prompt = conv.to_openai_api_messages()
        stream_iter = bard_api_stream_iter(
            model_api_dict["model_name"],
            prompt,
            temperature,
            top_p,
            api_key=model_api_dict["api_key"],
        )
    elif model_api_dict["api_type"] == "mistral":
        prompt = conv.to_openai_api_messages()
        stream_iter = mistral_api_stream_iter(
            model_name, prompt, temperature, top_p, max_new_tokens
        )
    elif model_api_dict["api_type"] == "nvidia":
        prompt = conv.to_openai_api_messages()
        stream_iter = nvidia_api_stream_iter(
            model_name,
            prompt,
            temperature,
            top_p,
            max_new_tokens,
            model_api_dict["api_base"],
        )
    elif model_api_dict["api_type"] == "ai2":
        prompt = conv.to_openai_api_messages()
        stream_iter = ai2_api_stream_iter(
            model_name,
            model_api_dict["model_name"],
            prompt,
            temperature,
            top_p,
            max_new_tokens,
            api_base=model_api_dict["api_base"],
            api_key=model_api_dict["api_key"],
        )
    else:
        raise NotImplementedError

    conv.update_last_message("▌")
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        for i, data in enumerate(stream_iter):
            if data["error_code"] == 0:
                output = data["text"].strip()
                conv.update_last_message(output + "▌")
                yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
            else:
                output = data["text"] + f"\n\n(error_code: {data['error_code']})"
                conv.update_last_message(output)
                yield (state, state.to_gradio_chatbot()) + (
                    disable_btn,
                    disable_btn,
                    disable_btn,
                    enable_btn,
                    enable_btn,
                )
                return
        output = data["text"].strip()
        if "vicuna" in model_name:
            output = post_process_code(output)
        conv.update_last_message(output)
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5
    except requests.exceptions.RequestException as e:
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n"
            f"(error_code: {ErrorCode.GRADIO_REQUEST_ERROR}, {e})"
        )
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return
    except Exception as e:
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n"
            f"(error_code: {ErrorCode.GRADIO_STREAM_UNKNOWN_ERROR}, {e})"
        )
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "gen_params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
            },
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")


block_css = """
#chatbot {
    line-height: 1.5;
}
#notice_markdown .prose {
    font-size: 120% !important;
}
#notice_markdown th {
    display: none;
}
#notice_markdown td {
    padding-top: 6px;
    padding-bottom: 6px;
}
#model_description_markdown {
    font-size: 120% !important;
}
#leaderboard_markdown .prose {
    font-size: 120% !important;
}
#leaderboard_markdown td {
    padding-top: 6px;
    padding-bottom: 6px;
}
#leaderboard_dataframe td {
    line-height: 0.1em;
}
#about_markdown .prose {
    font-size: 120% !important;
}
#ack_markdown .prose {
    font-size: 120% !important;
}
#input_box textarea {
}
footer {
    display:none !important;
}
.image-container {
    display: flex;
    align-items: center;
    padding: 1px;
}
.image-container img {
    margin: 0 30px;
    height: 30px;
    max-height: 100%;
    width: auto;
    max-width: 20%;
}
.image-about img {
    margin: 0 20px;
    margin-top: 20px;
    height: 40px;
    max-height: 100%;
    width: auto;
    float: left;
}
"""


def get_model_description_md(models):
    model_description_md = """
| | | |
| ---- | ---- | ---- |
"""
    ct = 0
    visited = set()
    for i, name in enumerate(models):
        minfo = get_model_info(name)
        if minfo.simple_name in visited:
            continue
        visited.add(minfo.simple_name)
        one_model_md = f"[{minfo.simple_name}]({minfo.link}): {minfo.description}"

        if ct % 3 == 0:
            model_description_md += "|"
        model_description_md += f" {one_model_md} |"
        if ct % 3 == 2:
            model_description_md += "\n"
        ct += 1
    return model_description_md


def build_about():
    about_markdown = """
# About Us
Chatbot Arena is an open-source research project developed by members from [LMSYS](https://lmsys.org/about/) and UC Berkeley [SkyLab](https://sky.cs.berkeley.edu/).  Our mission is to build an open crowdsourced platform to collect human feedback and evaluate LLMs under real-world scenarios. We open-source our [FastChat](https://github.com/lm-sys/FastChat) project at GitHub and release chat and human feedback datasets [here](https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md). We invite everyone to join us in this journey!

## Read More
- Chatbot Arena [launch post](https://lmsys.org/blog/2023-05-03-arena/), [data release](https://lmsys.org/blog/2023-07-20-dataset/)
- LMSYS-Chat-1M [report](https://arxiv.org/abs/2309.11998)

## Core Members
[Lianmin Zheng](https://lmzheng.net/), [Wei-Lin Chiang](https://infwinston.github.io/), [Ying Sheng](https://sites.google.com/view/yingsheng/home), [Siyuan Zhuang](https://scholar.google.com/citations?user=KSZmI5EAAAAJ)

## Advisors
[Ion Stoica](http://people.eecs.berkeley.edu/~istoica/), [Joseph E. Gonzalez](https://people.eecs.berkeley.edu/~jegonzal/), [Hao Zhang](https://cseweb.ucsd.edu/~haozhang/)

## Contact Us
- Follow our [Twitter](https://twitter.com/lmsysorg), [Discord](https://discord.gg/HSWAKCrnFx) or email us at lmsys.org@gmail.com
- File issues on [GitHub](https://github.com/lm-sys/FastChat)
- Download our datasets and models on [HuggingFace](https://huggingface.co/lmsys)

## Acknowledgment
We thank [SkyPilot](https://github.com/skypilot-org/skypilot) and [Gradio](https://github.com/gradio-app/gradio) team for their system support.
We also thank [Kaggle](https://www.kaggle.com/), [MBZUAI](https://mbzuai.ac.ae/), [a16z](https://www.a16z.com/), [Together AI](https://www.together.ai/), [Anyscale](https://www.anyscale.com/), [HuggingFace](https://huggingface.co/) for their generous sponsorship. Learn more about partnership [here](https://lmsys.org/donations/).

<div class="image-about">
    <img src="https://storage.googleapis.com/public-arena-asset/kaggle.png" alt="Kaggle">
    <img src="https://storage.googleapis.com/public-arena-asset/mbzuai.jpeg" alt="MBZUAI">
    <img src="https://storage.googleapis.com/public-arena-asset/a16z.jpeg" alt="a16z">
    <img src="https://storage.googleapis.com/public-arena-asset/together.png" alt="Together AI">
    <img src="https://storage.googleapis.com/public-arena-asset/anyscale.png" alt="AnyScale">
    <img src="https://storage.googleapis.com/public-arena-asset/huggingface.png" alt="HuggingFace">
</div>
"""

    # state = gr.State()
    gr.Markdown(about_markdown, elem_id="about_markdown")

    # return [state]


def build_single_model_ui(models, add_promotion_links=False):
    promotion = (
        """
- | [GitHub](https://github.com/lm-sys/FastChat) | [Dataset](https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/HSWAKCrnFx) |
- Introducing Llama 2: The Next Generation Open Source Large Language Model. [[Website]](https://ai.meta.com/llama/)
- Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality. [[Blog]](https://lmsys.org/blog/2023-03-30-vicuna/)

## 🤖 Choose any model to chat
"""
        if add_promotion_links
        else ""
    )

    notice_markdown = f"""
# 🏔️ Chat with Open Large Language Models
{promotion}
"""

    state = gr.State()
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Group(elem_id="share-region-named"):
        with gr.Row(elem_id="model_selector_row"):
            model_selector = gr.Dropdown(
                choices=models,
                value=models[0] if len(models) > 0 else "",
                interactive=True,
                show_label=False,
                container=False,
            )
        with gr.Row():
            with gr.Accordion(
                f"🔍 Expand to see the descriptions of {len(models)} models",
                open=False,
                elem_id="model_description_accordion",
            ):
                model_description_md = get_model_description_md(models)
                gr.Markdown(model_description_md, elem_id="model_description_markdown")

        chatbot = gr.Chatbot(
            elem_id="chatbot",
            label="Scroll down and start chatting",
            height=550,
            show_copy_button=True,
        )
    with gr.Row():
        textbox = gr.Textbox(
            show_label=False,
            placeholder="👉 Enter your prompt and press ENTER",
            elem_id="input_box",
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row() as button_row:
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    with gr.Accordion("Parameters", open=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=2048,
            value=1024,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    if add_promotion_links:
        gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    # Register listeners
    btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
    upvote_btn.click(
        upvote_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    downvote_btn.click(
        downvote_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    flag_btn.click(
        flag_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    regenerate_btn.click(regenerate, state, [state, chatbot, textbox] + btn_list).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    clear_btn.click(clear_history, None, [state, chatbot, textbox] + btn_list)

    model_selector.change(clear_history, None, [state, chatbot, textbox] + btn_list)

    textbox.submit(
        add_text, [state, model_selector, textbox], [state, chatbot, textbox] + btn_list
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    send_btn.click(
        add_text,
        [state, model_selector, textbox],
        [state, chatbot, textbox] + btn_list,
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )

    return [state, model_selector]


def build_demo(models):
    with gr.Blocks(
        title="Chat with Open Large Language Models",
        theme=gr.themes.Default(),
        css=block_css,
    ) as demo:
        url_params = gr.JSON(visible=False)

        state, model_selector = build_single_model_ui(models)

        if args.model_list_mode not in ["once", "reload"]:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

        if args.show_terms_of_use:
            load_js = get_window_url_params_with_tos_js
        else:
            load_js = get_window_url_params_js

        demo.load(
            load_demo,
            [url_params],
            [
                state,
                model_selector,
            ],
            js=load_js,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to generate a public, shareable link",
    )
    parser.add_argument(
        "--controller-url",
        type=str,
        default="http://localhost:21001",
        help="The address of the controller",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=10,
        help="The concurrency count of the gradio queue",
    )
    parser.add_argument(
        "--model-list-mode",
        type=str,
        default="once",
        choices=["once", "reload"],
        help="Whether to load the model list once or reload the model list every time",
    )
    parser.add_argument(
        "--moderate",
        action="store_true",
        help="Enable content moderation to block unsafe inputs",
    )
    parser.add_argument(
        "--show-terms-of-use",
        action="store_true",
        help="Shows term of use before loading the demo",
    )
    parser.add_argument(
        "--register-api-endpoint-file",
        type=str,
        help="Register API-based model endpoints from a JSON file",
    )
    parser.add_argument(
        "--gradio-auth-path",
        type=str,
        help='Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"',
    )
    parser.add_argument(
        "--gradio-root-path",
        type=str,
        help="Sets the gradio root path, eg /abc/def. Useful when running behind a reverse-proxy or at a custom URL path prefix",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Set global variables
    set_global_vars(args.controller_url, args.moderate)
    models, all_models = get_model_list(
        args.controller_url,
        args.register_api_endpoint_file,
    )

    # Set authorization credentials
    auth = None
    if args.gradio_auth_path is not None:
        auth = parse_gradio_auth_creds(args.gradio_auth_path)

    # Launch the demo
    demo = build_demo(models)
    demo.queue(
        default_concurrency_limit=args.concurrency_count,
        status_update_rate=10,
        api_open=False,
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
        auth=auth,
        root_path=args.gradio_root_path,
    )
