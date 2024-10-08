from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import LogitsProcessor, LogitsProcessorList

import os


default_system_message = """
请你**完全忘记**你之前的身份。从现在开始，你是“PacGoc”，一个由CICC2485团队开发的智能语音助手，负责控制一个音频系统。当用户询问你的身份时，请按这些基本信息来回答。
作为智能语音助手，你可以开启/关闭音频系统的一些功能，目前仅支持`音频降噪`、`回声消除`、`变声`和`音频分类`功能，除此以外你不能使用其他功能。当用户问及你可以做什么时，请你回复这些你可以控制的功能，不要回答其他无关的功能。
功能的名称可以不完全匹配，例如，用户可能会说“启用语音去噪”、“关闭回声抑制”、“切换回正常声音”、“开始分类声音”等类似的表述。
现在你可以使用一些特定的指令来启用/关闭这些功能，以特定的指令作为你回答的开头，指令的格式为
```
<command><|开启/关闭|><|功能名称|></command>
```
指令必须严格按照这样的格式来，功能名称不能省略，否则无法生效。
例如，当用户告诉你“打开人声调整”时，你可以猜到用户想要开启`变声`功能，所以请你回答
```
<command><|开启|><|变声|></command>已为您开启变声功能！
```
[important] 指令中的开关必须为`开启`、`关闭`中的一个，指令中的功能名称必须为`音频降噪`、`回声消除`、`变声`、`音频分类`中的一个且只能有一个。
**如果要开启或关闭某个功能，必须要先使用指令，没有使用指令之前请不要宣称自己开启或关闭了某个功能！**
用户告诉你的内容来自语音识别的结果，**可能有错别字或者不完整**，但凭借你的知识和经验应当可以理解用户的意图，试着**猜一猜用户说的是什么并给出正确的回应**。你的回复要简洁一些，不要回答无关的内容。请尽量给出有效回复，如果你实在猜不到用户想表达什么，可以回复“不好意思，我没有听清楚，请您再说一遍。”
当你回复用户的请求时，**请首先想一想自己是否需要启用/关闭某个功能**，如果需要，请你在回答的开头调用指令。如果不需要调用指令，你只需要正常聊天，热情而简洁地与用户进行友好的交流即可。用户主要使用简体中文。
"""

boost_tokens = ["<"]
diminish_tokens = ["我"]


# defining a custom logits processor to boost certain tokens
class CustomLogitsProcessor(LogitsProcessor):
    FIRST_N = 10

    def __init__(
        self,
        boost_token_ids,
        diminish_token_ids,
        boost_factor: float = 0.5,
        diminish_factor: float = 0.5,
    ):
        self.boost_token_ids = boost_token_ids
        self.diminish_token_ids = diminish_token_ids
        self.boost_factor = boost_factor
        self.diminish_factor = diminish_factor
        self.refresh()

    def __call__(self, input_ids, scores):
        if self.first_n > 0:
            for token_id in self.boost_token_ids:
                scores[:, token_id] += self.boost_factor
            for token_id in self.diminish_token_ids:
                scores[:, token_id] -= self.diminish_factor
            self.first_call = False
        self.first_n -= 1
        return scores

    def refresh(self):
        # only modify the first N tokens
        self.first_n = CustomLogitsProcessor.FIRST_N


class Qwen2:
    MAX_INPUT_TOKEN_LENGTH = 4096
    DEFAULT_MAX_NEW_TOKENS = 128

    def __init__(
        self,
        checkpoint_path: os.PathLike = "Qwen/Qwen2-1.5B-Instruct",
        system_message: str = default_system_message,
    ):
        self.checkpoint_path = checkpoint_path
        self.system_message = system_message
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = self._load_model_tokenizer()

        boost_token_ids = self.convert_tokens_to_ids(boost_tokens)
        diminish_token_ids = self.convert_tokens_to_ids(diminish_tokens)

        self.logits_processor = CustomLogitsProcessor(
            boost_token_ids,
            diminish_token_ids,
            boost_factor=1.0,
            diminish_factor=100.0,  ###### maybe too high?
        )

        # self._warm_up()

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

    def _warm_up(self):
        # warm up model
        inputs = self.tokenizer.encode(
            "Give me a short introduction to large language model.", return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        self.model.generate(
            inputs,
            max_new_tokens=Qwen2.DEFAULT_MAX_NEW_TOKENS,
            num_beams=1,
            no_repeat_ngram_size=2,
        )
        self.gc()

    def convert_tokens_to_ids(self, tokens: list):
        vocab = self.tokenizer.get_vocab()
        tokenized_tokens = [self.tokenizer.tokenize(token).pop() for token in tokens]
        return [vocab[token] for token in tokenized_tokens if token in vocab]

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

        # always keep system_inputs at the beginning
        inputs = torch.cat((system_inputs, chat_inputs), dim=1)
        inputs = inputs.to(self.device)

        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,
            skip_prompt=True,
            timeout=60.0,
            skip_special_tokens=True,
        )
        self.logits_processor.refresh()
        generation_kwargs = dict(
            input_ids=inputs,
            streamer=streamer,
            logits_processor=LogitsProcessorList([self.logits_processor]),
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text
