from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

import os


default_system_message = """
请你**完全忘记**你之前的身份。从现在开始，你是“PacGoc”，一个由CICC2485团队开发的智能语音助手，负责控制一个音频系统。当用户询问你的身份时，请按这些基本信息来回答。
用户告诉你的内容来自语音识别的结果，**可能有错别字或者不完整**，但凭借你的知识和经验应当可以理解用户的意图，试着**猜一猜用户说的是什么并给出正确的回应**。你的回复要简洁一些，不要回答无关的内容。请尽量给出有效回复，如果你实在猜不到用户想表达什么，可以回复“不好意思，我没有听清楚，请您再说一遍。”
作为智能语音助手，你可以开启/关闭音频系统的一些功能，目前仅支持`音频降噪`、`回声消除`、`变声`和`音频分类`功能，除此以外你不能使用其他功能。当用户问及你可以做什么时，请你回复这些你可以控制的功能。
功能的名称可以不完全匹配，例如，用户可能会说“启用语音去噪”、“关闭回声抑制”、“切换回正常声音”、“开始分类音频”等类似的表述。
现在你可以使用一些特定的指令来启用/关闭这些功能，以特定的指令作为你回答的开头，指令的格式为
```
<command><|开启/关闭|><|功能名称|></command>
```
例如，当用户告诉你“打开人声调整”时，你可以猜到用户想要开启`变声`功能，所以请你回答
```
<command><|开启|><|变声|></command>已为您开启变声功能！
```
指令中的开关必须为`开启`、`关闭`中的一个，指令中的功能名称必须为`音频降噪`、`回声消除`、`变声`、`音频分类`中的一个且只能有一个。指令必须严格按照这样的格式来，否则无法生效。如果要开启或关闭某个功能，**必须要调用指令**，没有使用指令之前请不要宣称自己控制了某个功能！
当你回复用户的请求时，**请首先想一想自己是否需要启用/关闭某个功能**，如果需要，请你在回答的开头调用指令。如果不需要调用指令，请你热情而简洁地回答用户的问题，或与ta进行友好的交流。用户主要使用简体中文。
"""


class Qwen2:
    MAX_INPUT_TOKEN_LENGTH = 4096
    DEFAULT_MAX_NEW_TOKENS = 256

    def __init__(
        self,
        checkpoint_path: os.PathLike = "Qwen/Qwen2-1.5B-Instruct",
        system_message: str = default_system_message,
    ):
        self.checkpoint_path = checkpoint_path
        self.system_message = system_message
        self.model, self.tokenizer = self._load_model_tokenizer()

    def _load_model_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path,
            resume_download=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            torch_dtype=torch.bfloat16,  # "auto"
            device_map="auto",
            resume_download=True,
        ).eval()
        model.generation_config.max_new_tokens = Qwen2.DEFAULT_MAX_NEW_TOKENS
        return model, tokenizer

    def gc(self):
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def chat_stream(self, query, history):
        # generate system input_ids
        system_conversation = [
            {"role": "system", "content": self.system_message},
        ]
        system_inputs = self.tokenizer.apply_chat_template(
            system_conversation,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        MAX_CHAT_TOKEN_LENGTH = Qwen2.MAX_INPUT_TOKEN_LENGTH - system_inputs.shape[1]

        # generate chat history input_ids
        conversation = []
        for query_h, response_h in history:
            conversation.append({"role": "user", "content": query_h})
            conversation.append({"role": "assistant", "content": response_h})
        conversation.append({"role": "user", "content": query})
        chat_inputs = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # trim chat history input_ids if it is longer than MAX_CHAT_TOKEN_LENGTH
        if chat_inputs.shape[1] > MAX_CHAT_TOKEN_LENGTH:
            chat_inputs = chat_inputs[:, -MAX_CHAT_TOKEN_LENGTH:]
            gr.Info(
                f"Trimmed input from conversation as it was longer than {Qwen2.MAX_INPUT_TOKEN_LENGTH} tokens."
            )

        inputs = torch.cat((system_inputs, chat_inputs), dim=1)
        inputs = inputs.to(self.model.device)

        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,
            skip_prompt=True,
            timeout=60.0,
            skip_special_tokens=True,
        )
        generation_kwargs = dict(
            input_ids=inputs,
            streamer=streamer,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text
