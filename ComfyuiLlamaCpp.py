from __future__ import annotations

import base64
import copy
import json
import os
import random
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from io import BytesIO
from pprint import pprint
from typing import Any

import numpy as np
from aiohttp import web
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from server import PromptServer

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
DEFAULT_MEDIA_MARKER = "<__media__>"


@dataclass
class ChatSession:
    messages: list[dict] = field(default_factory=list)


CHAT_SESSIONS: dict[str, ChatSession] = {}
NODE_DIR = os.path.dirname(os.path.realpath(__file__))
CONTEXT_DIR = os.path.join(NODE_DIR, "saved_context")
HISTORY_DIR = os.path.join(NODE_DIR, "saved_history")


def _safe_filename(name: str) -> str:
    name = os.path.basename(str(name).strip())
    if not name or name in (".", ".."):
        raise Exception("filename is empty")
    return name


def _list_saved_files(directory: str, suffix: str = "") -> list[str]:
    os.makedirs(directory, exist_ok=True)
    files = []
    for name in os.listdir(directory):
        if name == ".keep":
            continue
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        if suffix and not name.endswith(suffix):
            continue
        files.append(name)
    files.sort()
    return files


def _file_mtime(path: str) -> str:
    try:
        return str(os.path.getmtime(path))
    except OSError:
        return ""


def _error_message(body: Any) -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)
        if err:
            return str(err)
        return str(body)
    return str(body)


def _http_json(method: str, base_url: str, path: str, payload: dict | None = None, api_key: str = "", timeout: float | None = None) -> Any:
    url = base_url.rstrip("/") + path
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            msg = _error_message(json.loads(body))
        except Exception:
            msg = body[:500] or str(e.reason)
        raise Exception(f"llama-server {e.code} {path}: {msg}") from None
    except urllib.error.URLError as e:
        raise Exception(f"llama-server unreachable at {url}: {e.reason}") from None


def _api_key(connectivity: dict | None) -> str:
    if not connectivity:
        return ""
    return connectivity.get("api_key") or ""


def list_models(url: str, api_key: str = "") -> list[str]:
    data = _http_json("GET", url, "/v1/models", api_key=api_key, timeout=10)
    models = []
    if isinstance(data, dict):
        for item in data.get("data") or data.get("models") or []:
            if isinstance(item, dict):
                name = item.get("id") or item.get("name") or item.get("model")
                if name:
                    models.append(name)
            elif isinstance(item, str):
                models.append(item)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                name = item.get("id") or item.get("name")
                if name:
                    models.append(name)
            elif isinstance(item, str):
                models.append(item)
    return models


def _filter_enabled_options(options: dict[str, Any] | None, for_chat: bool = False) -> dict[str, Any] | None:
    if not options:
        return None
    out: dict[str, Any] = {}
    for key, enabled in options.items():
        if not key.startswith("enable_") or not enabled:
            continue
        name = key[len("enable_"):]
        value = options.get(name)
        if name == "stop":
            if isinstance(value, str):
                parts = [p for p in value.split("\n") if p != ""]
                value = parts if parts else ([value] if value else [])
            elif not isinstance(value, list):
                value = [value]
        if name == "num_predict":
            out["max_tokens" if for_chat else "n_predict"] = value
        else:
            out[name] = value
    return out or None


def _images_to_b64(images) -> list[str]:
    images_b64 = []
    for image in images:
        i = 255.0 * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        images_b64.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
    return images_b64


def _parse_context(context) -> list[int] | None:
    if context is None:
        return None
    if isinstance(context, str):
        if not context.strip():
            return None
        return [int(item.strip()) for item in context.split(",") if item.strip() != ""]
    if isinstance(context, list):
        return [int(item) for item in context]
    return None


def _split_thinking(text: str | None, think: bool, extra: str | None = None) -> tuple[str | None, str | None]:
    if not think:
        return text, None
    if extra:
        return text, extra
    if not text:
        return text, None
    match = THINK_RE.search(text)
    if not match:
        return text, None
    thinking = match.group(1).strip()
    result = (text[:match.start()] + text[match.end():]).strip()
    return result, thinking


def _apply_template(url: str, api_key: str, messages: list[dict], think: bool) -> str:
    payload = {
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": think},
    }
    try:
        data = _http_json("POST", url, "/apply-template", payload, api_key=api_key)
        prompt = data.get("prompt")
        if isinstance(prompt, str) and prompt:
            return prompt
    except Exception:
        pass
    parts = []
    for message in messages:
        parts.append(f"{message.get('role', 'user')}: {message.get('content', '')}")
    parts.append("assistant:")
    return "\n".join(parts)


def _tokenize(url: str, api_key: str, content: str) -> list[int]:
    data = _http_json("POST", url, "/tokenize", {"content": content}, api_key=api_key)
    tokens = data.get("tokens") or []
    out = []
    for token in tokens:
        if isinstance(token, int):
            out.append(token)
        elif isinstance(token, dict) and "id" in token:
            out.append(int(token["id"]))
    return out


def _media_info(url: str, api_key: str, model: str) -> tuple[bool, str]:
    path = "/props"
    if model:
        path = "/props?model=" + urllib.parse.quote(model)
    try:
        props = _http_json("GET", url, path, api_key=api_key, timeout=10)
    except Exception:
        return False, DEFAULT_MEDIA_MARKER
    modalities = props.get("modalities") or {}
    vision = bool(modalities.get("vision"))
    marker = props.get("media_marker") or DEFAULT_MEDIA_MARKER
    return vision, marker


def _think_payload(think: bool) -> dict:
    if think:
        return {"chat_template_kwargs": {"enable_thinking": True}}
    return {
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
    }


def _debug_enabled(options: dict | None) -> bool:
    return bool(options and options.get("debug"))


@PromptServer.instance.routes.post("/llamacpp/get_models")
async def get_models_endpoint(request):
    data = await request.json()
    url = data.get("url")
    api_key = data.get("api_key") or ""
    try:
        models = list_models(url, api_key)
        return web.json_response(models)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


class LlamaCppSaveContext:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("LLAMACPP_CONTEXT", {"forceInput": True, "tooltip": "Connect the context output from LlamaCpp Generate."}),
                "filename": ("STRING", {"default": "context"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "llamacpp_save_context"
    OUTPUT_NODE = True
    CATEGORY = "LlamaCpp"
    DESCRIPTION = "Save Generate context token ids to saved_context/."

    def llamacpp_save_context(self, filename, context=None):
        os.makedirs(CONTEXT_DIR, exist_ok=True)
        filename = _safe_filename(filename)
        if not filename.endswith(".png"):
            filename = filename + ".png"
        metadata = PngInfo()
        if isinstance(context, list):
            metadata.add_text("context", ",".join(map(str, context)))
        else:
            metadata.add_text("context", str(context) if context is not None else "")
        image = Image.new("RGB", (100, 100), (255, 255, 255))
        image.save(os.path.join(CONTEXT_DIR, filename), pnginfo=metadata)
        return {"ui": {"context": context}}


class LlamaCppLoadContext:
    @classmethod
    def INPUT_TYPES(s):
        files = _list_saved_files(CONTEXT_DIR, ".png")
        return {
            "required": {
                "context_file": (files or [""], {"tooltip": "A previously saved Generate context file."}),
            },
        }

    CATEGORY = "LlamaCpp"
    RETURN_NAMES = ("context",)
    RETURN_TYPES = ("LLAMACPP_CONTEXT",)
    FUNCTION = "llamacpp_load_context"
    DESCRIPTION = "Load saved Generate context into LlamaCpp Generate."

    @classmethod
    def IS_CHANGED(s, context_file):
        return _file_mtime(os.path.join(CONTEXT_DIR, context_file))

    def llamacpp_load_context(self, context_file):
        path = os.path.join(CONTEXT_DIR, _safe_filename(context_file))
        with Image.open(path) as img:
            res = img.info.get("context", "")
        return (res,)


class LlamaCppSaveHistory:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "history": ("LLAMACPP_HISTORY", {"forceInput": True, "tooltip": "Connect the history output from LlamaCpp Chat."}),
                "filename": ("STRING", {"default": "history"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "llamacpp_save_history"
    OUTPUT_NODE = True
    CATEGORY = "LlamaCpp"
    DESCRIPTION = "Save Chat message history to saved_history/ as JSON."

    def llamacpp_save_history(self, filename, history=None):
        os.makedirs(HISTORY_DIR, exist_ok=True)
        filename = _safe_filename(filename)
        if not filename.endswith(".json"):
            filename = filename + ".json"
        session = CHAT_SESSIONS.get(history) if history is not None else None
        messages = session.messages if session is not None else []
        path = os.path.join(HISTORY_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        return {"ui": {"history": history, "filename": filename}}


class LlamaCppLoadHistory:
    @classmethod
    def INPUT_TYPES(s):
        files = _list_saved_files(HISTORY_DIR, ".json")
        return {
            "required": {
                "history_file": (files or [""], {"tooltip": "A previously saved Chat history file."}),
            },
        }

    CATEGORY = "LlamaCpp"
    RETURN_NAMES = ("history",)
    RETURN_TYPES = ("LLAMACPP_HISTORY",)
    FUNCTION = "llamacpp_load_history"
    DESCRIPTION = "Load saved Chat history into LlamaCpp Chat."

    @classmethod
    def IS_CHANGED(s, history_file):
        return _file_mtime(os.path.join(HISTORY_DIR, history_file))

    def llamacpp_load_history(self, history_file):
        path = os.path.join(HISTORY_DIR, _safe_filename(history_file))
        with open(path, encoding="utf-8") as f:
            messages = json.load(f)
        if not isinstance(messages, list):
            raise Exception("saved history is not a message list")
        key = os.path.splitext(os.path.basename(history_file))[0]
        CHAT_SESSIONS[key] = ChatSession(messages=messages)
        return (key,)


class LlamaCppOptions:
    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "enable_mirostat": ("BOOLEAN", {"default": False}),
                "mirostat": ("INT", {"default": 0, "min": 0, "max": 2, "step": 1, "tooltip": "Whether to use Mirostat sampling. Mirostat is an algorithm that actively maintains the quality of generated text within a desired range during text generation. (0 = disabled, 1 = Mirostat 1, 2 = Mirostat 2.0)"}),
                "enable_mirostat_eta": ("BOOLEAN", {"default": False}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0, "step": 0.1, "tooltip": "Mirostat's learning rate parameter influences how quickly the algorithm responds to feedback from the generated text."}),
                "enable_mirostat_tau": ("BOOLEAN", {"default": False}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0, "step": 0.1, "tooltip": "Mirostat's target entropy parameter controls the balance between coherence and diversity in the generated text."}),
                "enable_repeat_last_n": ("BOOLEAN", {"default": False}),
                "repeat_last_n": ("INT", {"default": 64, "min": -1, "max": 64, "step": 1, "tooltip": "Sets how far back for the model to look back to prevent repetition. (0 = disabled, -1 = context size)"}),
                "enable_repeat_penalty": ("BOOLEAN", {"default": False}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0, "max": 2, "step": 0.05, "tooltip": "Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient."}),
                "enable_temperature": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.8, "min": -10, "max": 10, "step": 0.05, "tooltip": "Increasing the temperature will make the model answer more creatively."}),
                "enable_seed": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1, "tooltip": "Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt."}),
                "enable_stop": ("BOOLEAN", {"default": False}),
                "stop": ("STRING", {"default": "", "multiline": False, "tooltip": "When this pattern is encountered the LLM will stop generating text and return. Separate multiple stops with newlines."}),
                "enable_num_predict": ("BOOLEAN", {"default": False}),
                "num_predict": ("INT", {"default": -1, "min": -1, "max": 2048, "step": 1, "tooltip": "Maximum number of tokens to predict when generating text. The default -1 means infinite generation."}),
                "enable_top_k": ("BOOLEAN", {"default": False}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1, "tooltip": "Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative."}),
                "enable_top_p": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05, "tooltip": "Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text."}),
                "enable_min_p": ("BOOLEAN", {"default": False}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05, "tooltip": "Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter p represents the minimum probability for a token to be considered, relative to the probability of the most likely token."}),
                "debug": ("BOOLEAN", {"default": False, "tooltip": "Print request and response details to the ComfyUI console. Not sent to llama-server."}),
            },
        }

    RETURN_TYPES = ("LLAMACPP_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "llamacpp_options"
    CATEGORY = "LlamaCpp"
    DESCRIPTION = "Sampling options for llama-server. Enable a value for it to be sent on Generate or Chat."

    def llamacpp_options(self, **kargs):
        if kargs["debug"]:
            print("--- llamacpp options dump\n")
            pprint(kargs)
            print("---------------------------------------------------------")
        return (kargs,)


class LlamaCppConnectivity:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:8080",
                    "tooltip": "The URL of llama-server. Default is a local instance on port 8080.",
                }),
                "model": ((), {"tooltip": "Select a model reported by llama-server. If this list is empty, start llama-server and press Reconnect."}),
            },
            "optional": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Bearer token if llama-server was started with --api-key.",
                }),
            },
        }

    RETURN_TYPES = ("LLAMACPP_CONNECTIVITY",)
    RETURN_NAMES = ("connection",)
    FUNCTION = "llamacpp_connectivity"
    CATEGORY = "LlamaCpp"
    DESCRIPTION = "Connection to llama-server. Use Reconnect to load the model list."

    def llamacpp_connectivity(self, url, model, api_key=""):
        data = {
            "url": url,
            "model": model,
            "api_key": api_key,
        }
        return (data,)


class LlamaCppGenerate:
    def __init__(self):
        self.saved_context = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are an AI artist.",
                    "tooltip": "System prompt - use this to set the role and general behavior of the model.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "What is art?",
                    "tooltip": "User prompt - a question or task you want the model to answer or perform. For vision tasks, you can refer to the input image as 'this image', 'photo' etc. like 'Describe this image in detail'",
                }),
                "think": ("BOOLEAN", {"default": False, "tooltip": "If enabled, the model will do a thinking process before answering. The thinking is then available as a separate output. Some models don't support this feature."}),
                "keep_context": ("BOOLEAN", {"default": False, "tooltip": "If enabled, the model will keep the context of the conversation and use it for the next generation."}),
                "format": (["text", "json"], {"tooltip": "Output format of the response. 'text' will return a plain text response, while 'json' will constrain the model to JSON."}),
            },
            "optional": {
                "connectivity": ("LLAMACPP_CONNECTIVITY", {"forceInput": False, "tooltip": "Set a llama-server provider for the generation. If this input is empty, the 'meta' input must be set."}),
                "options": ("LLAMACPP_OPTIONS", {"forceInput": False, "tooltip": "Connect a LlamaCpp Options node for advanced inference configuration."}),
                "images": ("IMAGE", {"forceInput": False, "tooltip": "Provide an image or a batch of images for vision tasks. llama-server must be started with --mmproj."}),
                "context": ("LLAMACPP_CONTEXT", {"forceInput": False, "tooltip": "Optionally set an existing model context, useful for multi-turn conversations, follow-up questions."}),
                "meta": ("LLAMACPP_META", {"forceInput": False, "tooltip": "Use this input to chain multiple LlamaCpp Generate nodes. Connectivity and options are passed along."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "LLAMACPP_CONTEXT", "LLAMACPP_META")
    RETURN_NAMES = ("result", "thinking", "context", "meta")
    FUNCTION = "llamacpp_generate"
    CATEGORY = "LlamaCpp"
    DESCRIPTION = "Text generation with llama-server /completion. Supports vision, context chaining, and sampling options."

    def llamacpp_generate(self, system, prompt, think, keep_context, format, context=None, options=None, connectivity=None, images=None, meta=None):
        if connectivity is None and meta is None:
            raise Exception("Required input connectivity or meta.")
        if connectivity is None and meta["connectivity"] is None:
            raise Exception("Required input connectivity or connectivity in meta.")

        if meta is not None:
            if connectivity is not None:
                meta["connectivity"] = connectivity
            if options is not None:
                meta["options"] = options
        else:
            meta = {"options": options, "connectivity": connectivity}

        connectivity = meta["connectivity"]
        url = connectivity["url"]
        model = connectivity["model"]
        api_key = _api_key(connectivity)
        debug_print = _debug_enabled(meta.get("options"))
        request_options = _filter_enabled_options(meta.get("options"), for_chat=False)

        context = _parse_context(context)
        if keep_context and context is None:
            context = self.saved_context

        images_b64 = _images_to_b64(images) if images is not None else None
        user_content = prompt
        marker = None
        if images_b64:
            vision, marker = _media_info(url, api_key, model)
            if not vision:
                raise Exception("llama-server is not multimodal. Start it with --mmproj for vision models.")
            user_content = (marker * len(images_b64)) + "\n" + prompt

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})

        if context is not None:
            tokenize_text = user_content
            if images_b64:
                prompt_payload = list(context) + [{"prompt_string": user_content, "multimodal_data": images_b64}]
            else:
                prompt_payload = list(context) + [user_content]
        else:
            templated = _apply_template(url, api_key, messages, think)
            tokenize_text = templated
            if images_b64:
                prompt_payload = {"prompt_string": templated, "multimodal_data": images_b64}
            else:
                prompt_payload = templated

        body = {
            "model": model,
            "prompt": prompt_payload,
            "cache_prompt": True,
            "return_tokens": True,
            "stream": False,
        }
        body.update(_think_payload(think))
        if request_options:
            body.update(request_options)
        if format == "json":
            body["json_schema"] = {}

        if debug_print:
            print(f"""
--- llamacpp generate request:

url: {url}
model: {model}
system: {system}
prompt: {prompt}
images: {0 if images_b64 is None else len(images_b64)}
context: {context}
think: {think}
options: {request_options}
format: {format}
---------------------------------------------------------
""")

        response = _http_json("POST", url, "/completion", body, api_key=api_key)
        if isinstance(response, list):
            response = response[0] if response else {}

        if debug_print:
            print("\n--- llamacpp generate response:")
            pprint(response)
            print("---------------------------------------------------------")

        result_text = response.get("content")
        if result_text is None:
            result_text = response.get("response")
        extra_thinking = response.get("reasoning_content") or response.get("reasoning")
        result_text, thinking = _split_thinking(result_text, think, extra_thinking)

        generated_tokens = response.get("tokens") or []
        generated_tokens = [int(t) for t in generated_tokens if isinstance(t, (int, float))]
        prompt_tokens = []
        try:
            if context is not None:
                prompt_tokens = list(context) + _tokenize(url, api_key, tokenize_text)
            else:
                prompt_tokens = _tokenize(url, api_key, tokenize_text)
        except Exception:
            prompt_tokens = list(context) if context is not None else []
        new_context = prompt_tokens + generated_tokens

        if keep_context:
            self.saved_context = new_context
            if debug_print:
                print("saving context to node memory.")

        return result_text, thinking, new_context, meta


class LlamaCppChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are an AI artist.",
                    "tooltip": "System prompt - use this to set the role and general behavior of the model.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "What is art?",
                    "tooltip": "User prompt - a question or task you want the model to answer or perform. For vision tasks, you can refer to the input image as 'this image', 'photo' etc. like 'Describe this image in detail'",
                }),
                "think": ("BOOLEAN", {"default": False, "tooltip": "If enabled, the model will do a thinking process before answering. The thinking is then available as a separate output."}),
                "format": (["text", "json"], {"tooltip": "Output format of the response. 'text' will return a plain text response, while 'json' will constrain the model to JSON."}),
            },
            "optional": {
                "connectivity": ("LLAMACPP_CONNECTIVITY", {"forceInput": False, "tooltip": "Set a llama-server provider for the generation. If this input is empty, the 'meta' input must be set."}),
                "options": ("LLAMACPP_OPTIONS", {"forceInput": False, "tooltip": "Connect a LlamaCpp Options node for advanced inference configuration."}),
                "images": ("IMAGE", {"forceInput": False, "tooltip": "Provide an image or a batch of images for vision tasks. llama-server must be started with --mmproj."}),
                "meta": ("LLAMACPP_META", {"forceInput": False, "tooltip": "Use this input to chain multiple LlamaCpp Chat nodes. Connectivity and options are passed along."}),
                "history": ("LLAMACPP_HISTORY", {"forceInput": False, "tooltip": "Optionally set an existing chat history, useful for multi-turn conversations."}),
                "reset_session": ("BOOLEAN", {"default": False, "tooltip": "Clear the conversation history. WARNING: If using shared history, this will affect all nodes using the same history ID."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "LLAMACPP_META", "LLAMACPP_HISTORY")
    RETURN_NAMES = ("result", "thinking", "meta", "history")
    FUNCTION = "llamacpp_chat"
    CATEGORY = "LlamaCpp"
    DESCRIPTION = "Chat with llama-server /v1/chat/completions. Supports vision, history chaining, and sampling options."

    def llamacpp_chat(
        self,
        system,
        prompt,
        think,
        format,
        unique_id=None,
        options=None,
        connectivity=None,
        images=None,
        meta=None,
        history=None,
        reset_session=False,
    ):
        if meta is None:
            if connectivity is None:
                raise ValueError("Either 'connectivity' or 'meta' must be provided.")
            meta = {}

        if connectivity is not None:
            meta["connectivity"] = connectivity
        if options is not None:
            meta["options"] = options

        if "connectivity" not in meta or meta["connectivity"] is None:
            raise ValueError("'connectivity' must be present in meta.")

        connectivity = meta["connectivity"]
        url = connectivity["url"]
        model = connectivity["model"]
        api_key = _api_key(connectivity)
        debug_print = _debug_enabled(meta.get("options"))
        request_options = _filter_enabled_options(meta.get("options"), for_chat=True)

        images_b64 = _images_to_b64(images) if images is not None else None

        session_key = history if history is not None else unique_id
        if session_key is None:
            raise ValueError("Chat session key is missing.")

        if reset_session:
            CHAT_SESSIONS[session_key] = ChatSession()
            if debug_print:
                print(f"Session {session_key} has been reset")

        if session_key not in CHAT_SESSIONS:
            CHAT_SESSIONS[session_key] = ChatSession()

        session = CHAT_SESSIONS[session_key]
        history = session_key

        if system:
            if session.messages and session.messages[0].get("role") == "system":
                session.messages[0] = {"role": "system", "content": system}
            else:
                session.messages.insert(0, {"role": "system", "content": system})

        session.messages.append({"role": "user", "content": prompt})

        if debug_print:
            print(f"""
--- llamacpp chat request:

url: {url}
model: {model}
system: {system}
prompt: {prompt}
images: {0 if images_b64 is None else len(images_b64)}
think: {think}
options: {request_options}
format: {format}
---------------------------------------------------------
""")
            print("\n--- llamacpp chat session:")
            for message in session.messages:
                content = message.get("content", "")
                preview = content[:50] if isinstance(content, str) else str(content)[:50]
                pprint(f"{message['role']}> {preview}...")
            print("---------------------------------------------------------")

        messages_for_api = copy.deepcopy(session.messages)
        if images_b64 is not None:
            content = [{"type": "text", "text": prompt}]
            for b64 in images_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            messages_for_api[-1]["content"] = content

        body = {
            "model": model,
            "messages": messages_for_api,
            "stream": False,
        }
        body.update(_think_payload(think))
        if request_options:
            body.update(request_options)
        if format == "json":
            body["response_format"] = {"type": "json_object"}

        response = _http_json("POST", url, "/v1/chat/completions", body, api_key=api_key)

        if debug_print:
            print("\n--- llamacpp chat response:")
            pprint(response)
            print("---------------------------------------------------------")

        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        result_text = message.get("content")
        extra_thinking = message.get("reasoning_content") or message.get("reasoning")
        result_text, thinking = _split_thinking(result_text, think, extra_thinking)

        session.messages.append({
            "role": "assistant",
            "content": result_text or "",
        })

        return result_text, thinking, meta, history


NODE_CLASS_MAPPINGS = {
    "LlamaCppOptions": LlamaCppOptions,
    "LlamaCppConnectivity": LlamaCppConnectivity,
    "LlamaCppGenerate": LlamaCppGenerate,
    "LlamaCppSaveContext": LlamaCppSaveContext,
    "LlamaCppLoadContext": LlamaCppLoadContext,
    "LlamaCppChat": LlamaCppChat,
    "LlamaCppSaveHistory": LlamaCppSaveHistory,
    "LlamaCppLoadHistory": LlamaCppLoadHistory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCppOptions": "LlamaCpp Options",
    "LlamaCppConnectivity": "LlamaCpp Connectivity",
    "LlamaCppGenerate": "LlamaCpp Generate",
    "LlamaCppSaveContext": "LlamaCpp Save Context",
    "LlamaCppLoadContext": "LlamaCpp Load Context",
    "LlamaCppChat": "LlamaCpp Chat",
    "LlamaCppSaveHistory": "LlamaCpp Save History",
    "LlamaCppLoadHistory": "LlamaCpp Load History",
}
