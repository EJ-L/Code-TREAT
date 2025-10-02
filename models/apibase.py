from tenacity import retry, stop_after_attempt, wait_random_exponential
from zhipuai import ZhipuAI
import openai
from openai import OpenAI
import anthropic
import google.generativeai as genai
import time
import yaml

OFFICIAL_MODEL_NAME = {
    "grok-3-mini": "Grok-3-Mini (High)",
    "x-ai/grok-3-mini": "Grok-3-Mini (High)",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct": "Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama-3.3-70B-Instruct",
    "llama3.3:latest": "Llama3.3-70B-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/llama-4-scout:free": "Llama-4-Scout-17B-16E-Instruct",
    "deepseek-chat": "DeepSeek-V3 (0324)",
    "deepseek-reasoner": "DeepSeek-R1 (0528)",
    # "deepseek-r1-250528": "DeepSeek-R1 (0528)",
    "deepseek/deepseek-r1-0528": "DeepSeek-R1 (0528)",
    "deepseek-ai/DeepSeek-R1-0528": "DeepSeek-R1 (0528)",
    "deepseek-r1": "DeepSeek-R1",
    "deepseek/deepseek-r1": "DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1": "DeepSeek-R1",
    "deepseek/deepseek-r1:free": "DeepSeek-R1",
    "gpt-4.1-2025-04-14": "GPT-4.1-2025-04-14",
    "openai/gpt-4.1-2025-04-14": "GPT-4.1-2025-04-14",
    "claude-3-5-sonnet-20241022": "Claude-3.5-Sonnet-20241022",
    "claude-3-5-haiku-20241022": "Claude-3.5-Haiku-20241022",
    "anthropic/claude-3.5-haiku-20241022": "Claude-3.5-Haiku-20241022",
    "anthropic/claude-3.7-sonnet-latest": "Claude-3.7-Sonnet",
    "claude-3-7-sonnet-20250219": "Claude-3.7-Sonnet",
    "anthropic/claude-3.7-sonnet": "Claude-3.7-Sonnet",
    "claude-sonnet-4-20250514": "Claude-Sonnet-4",
    "anthropic/claude-sonnet-4": "Claude-Sonnet-4",
    "gpt-3.5-turbo-0125": "GPT-3.5-turbo-0125",
    "gpt-4o-2024-11-20": "GPT-4o-2024-11-20",
    "openai/gpt-4o-2024-11-20": "GPT-4o-2024-11-20",
    "gpt-4-turbo-2024-04-09": "GPT-4-turbo-2024-04-09",
    "gemma-2-9b-it": "Gemma-2-9B-Instruct",
    "gemma-2-27b-it": "Gemma-2-27B-Instruct",
    "google/gemma-3-27b-it": "Gemma-3-27B-Instruct",
    "o3-mini": "o3-mini (Med)",
    "o4-mini": "o4-mini (Med)",
    "openai/o4-mini-2025-04-16": "o4-mini (Med)",
    "gemini-1.5-flash-latest": "Gemini-1.5-Flash",
    "gemini-1.5-pro-latest": "Gemini-1.5-Pro",
    "gemini-2.0-flash-exp": "Gemini-2.0-Flash",
    "google/gemini-2.5-pro-preview-03-25": "Gemini-2.5-Pro-Preview-03-25",
    "google/gemini-2.5-pro-preview-05-06": "Gemini-2.5-Pro-Preview-05-06",
    "gemini-2.5-pro-preview-05-06": "Gemini-2.5-Pro-Preview-05-06",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct": "Qwen2.5-Coder-32B-Instruct",
    "Qwen/QwQ-32B": "QwQ-32B",
    "Qwen/Qwen3-30B-A3B": "Qwen3-30B-A3B",
    "Qwen/Qwen3-32B": "Qwen3-32B",
    "qwen3-32b": "Qwen3-32B",
    "Qwen/Qwen3-235B-A22B": "Qwen3-235B-A22B",
    "qwen/qwen3-235b-a22b-04-28": "Qwen3-235B-A22B",
    "qwen3:32b": "Qwen3-32B",
    "google/gemma-2-9b-it": "Gemma-2-Instruct",
    "google/gemma-2-27b-it": "Gemma-2-27B-Instruct",
    "anthropic/claude-3.5-sonnet": "Claude-3.5-Sonnet-20241022",
    "openai/gpt-4o-2024-11-20": "GPT-4o-2024-11-20",
    "openai/gpt-4.1-2025-04-14": "GPT-4.1-2025-04-14",
    "openai/o3-mini": "o3-mini (Med)",
    "openai/gpt-4-turbo": "GPT-4-turbo-2024-04-09",
    "openai/gpt-3.5-turbo-0125": "GPT-3.5-turbo-0125",
    "deepseek-ai/DeepSeek-V3": "DeepSeek-V3",
    "glm-4-flash": "GLM-4-Flash",
    "gpt-5": "GPT-5",
    "gpt-5-nano": "GPT-5-Nano",
    "gpt-5-mini": "GPT-5-Mini",
    "glm-4.5-flash": "GLM-4.5-Flash",
}


def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def load_model_specification(config):
    model_spec = config['model_specification']
    return model_spec.values()


config = load_yaml("configs/configs.yaml")
_, temperature, top_k, top_p = load_model_specification(config)

# Keep your existing dict but guard lookups with a default below
max_tokens_dict = {
    "o3-mini (Med)": 16384,
    "gemma-2-9b-it": 8192,
    "gemma-2-27b-it": 8192,
    "Llama-3.1-8B-Instruct": 8192,
    "Llama-3.1-70B-Instruct": 8192,
    "Llama-3.3-70B-Instruct": 16384,
    "Qwen2.5-72B-Instruct": 8192,
    "DeepSeek-V3": 8192,
    "gemini-1.5-flash-latest": 8192,
    "gemini-1.5-pro-latest": 8192,
    "gemini-2.0-flash-exp": 8192,
    "GPT-3.5-turbo-0125": 4096,
    "GPT-4o-2024-11-20": 16384,
    "GPT-4-turbo-2024-04-09": 4096,
    "meta-llama/Meta-Llama-3.3-70B-Instruct": 16384,
    "Qwen/QwQ-32B": 131072,
    "Gemma-3-27B-Instruct": 8192,
    "Claude-3.5-Sonnet-20241022": 8192,
    "Claude-3.5-Haiku-20241022": 8192,
    "deepseek-reasoner": 8192,
    "deepseek-chat": 8192,
    "Qwen2.5-Coder-32B-Instruct": 16384,
    "Claude-3.7-Sonnet": 16384,
    "GPT-4.1-2025-04-14": 16384,
    "Grok-3-Mini (High)": 65536,
    "gemini-2.5-pro-exp-03-25": 16384,
    "GLM-4-Flash": 16384,
    "GLM-4.5-Flash": 16384,
    "google/gemini-2.5-pro-preview": 16384,
    "Llama-4-Scout-17B-16E-Instruct": 16384,
    "DeepSeek-R1": 16384,
    "anthropic/claude-3.7-sonnet": 16384,
    "Qwen3-30B-A3B": 16384,
    "Qwen3-32B": 16384,
    "Qwen3-235B-A22B": 65536,
    "anthropic/claude-4-sonnet": 16384,
    "anthropic/claude-sonnet-4": 16384,
    "DeepSeek-R1 (0528)": 65536,
    "Gemini-2.5-Pro-Preview-05-06": 16384,
    "Claude-Sonnet-4": 16384,
    "o4-mini (Med)": 16384,
    "GPT-5": 32768,
    "GPT-5-Nano": 32768,
    "GPT-5-Mini": 32768,
}


delay_time = 1


class APIBase:
    def __init__(self, api_key_env, base_url, api_model_name, *, verbose: bool = False):
        self.api_key = api_key_env
        self.base_url = base_url
        self.api_model_name = api_model_name
        self.model_name = OFFICIAL_MODEL_NAME.get(api_model_name, api_model_name)
        self.client = None
        self.verbose = verbose

    def _log(self, *args):
        if self.verbose:
            print(*args)

    def setup_client(self, client_library):
        self.client_library = client_library
        try:
            if client_library in ["OPENAI_CLIENT", "CHATANYWHERE_CLIENT", "DEEPSEEK_CLIENT", "DEEPINFRA_CLIENT", "OLLAMA", "GROK_CLIENT"]:
                if client_library == "CHATANYWHERE_CLIENT":
                    openai.api_base = "https://api.chatanywhere.tech/v1"
                # Do NOT override global openai.api_base; rely on per-instance base_url
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            elif client_library == "OPENROUTER_CLIENT":
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    default_headers={"Accept-Encoding": "identity"},
                )
            elif client_library == "ZHIPU_CLIENT":
                self.client = ZhipuAI(api_key=self.api_key)
            elif client_library == "ANTHROPIC_CLIENT":
                self.client = anthropic.Anthropic(api_key=self.api_key)
            elif client_library == "GOOGLE_CLIENT":
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model_name)
            else:
                raise NotImplementedError(f"Client library not supported: {client_library}")
        except Exception as e:
            raise e

    @retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(6))
    def chat(self, messages, top_p=top_p, temperature=temperature, max_tokens=0, delay=delay_time, logprobs=False, **kwargs):
        n = kwargs.pop('n', 1)
        
        while True:
            # print(messages)
            
            try:
                # Resolve a safe max_tokens
                resolved_max_tokens = max_tokens_dict.get(self.model_name, max_tokens or 16384)

                self._log("SENDING REQUEST")
                self._log("MODEL NAME:", self.model_name)

                # OpenAI-compatible APIs
                if self.model_name.lower().startswith(("o3", "o4")):
                    response = self.client.chat.completions.create(
                        model=self.api_model_name,
                        messages=messages,
                        stream=False,
                        n=n,
                        **kwargs,
                    )
                elif self.model_name.lower().startswith("glm"):
                    response = self.client.chat.completions.create(
                        model=self.api_model_name,
                        messages=messages,
                        top_p=top_p,
                        temperature=temperature,
                        max_tokens=resolved_max_tokens,
                        stream=False,
                        **kwargs,
                    )
                elif self.model_name.lower().startswith(("grok-3-mini", "grok-3-mini-beta")):
                    response = self.client.chat.completions.create(
                        model=self.api_model_name,
                        messages=messages,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        max_tokens=resolved_max_tokens,
                        stream=False,
                        n=n,
                        reasoning_effort="high",
                        **kwargs,
                    )
                elif self.model_name.lower().startswith(("gpt-5",)):
                    response = self.client.chat.completions.create(
                        model=self.api_model_name,
                        messages=messages,
                        stream=False,
                        n=n,
                        **kwargs,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.api_model_name,
                        messages=messages,
                        top_p=top_p,
                        temperature=temperature,
                        top_k=top_k,
                        max_tokens=resolved_max_tokens,
                        stream=False,
                        n=n,
                        **kwargs,
                    )
                break

            except openai.BadRequestError as e:
                emsg = str(e)
                self._log("BAD REQUEST: ", emsg)
                if (
                    "context_length_exceeded" in getattr(e, "code", "")
                    or "context_length_exceeded" in emsg
                    or ("Error code: 400" in emsg and ("exceeds maximum input length" in emsg or "maximum context length is" in emsg))
                ):
                    return ["Exceeds Context Length"] * n
                # Generic 400 -> bubble up for visibility
                raise

            except openai.InternalServerError as e:
                self._log("INTERNAL SERVER ERROR", e)
                if "maximum context length is" in str(e):
                    return ["Exceeds Context Length"] * n
                # transient
                continue

            except openai.RateLimitError as e:
                self._log("RATE LIMITED", e)
                time.sleep(60)
                continue

            except Exception as e:
                print(e)
                emsg = str(e)
                self._log("ERROR MSG: ", emsg)

                if "context_length_exceeded" in getattr(e, "code", ""):
                    return ["Exceeds Context Length"] * n
                if "Connection Error" in emsg or "Error code: 500" in emsg:
                    continue
                # Bubble other errors to caller after logging
                raise

        # Unified return
        if n == 1:
            if logprobs:
                return [response.choices[0].message.content], [response.choices[0].logprobs.content[0].top_logprobs]
            else:
                self._log("response", getattr(response.choices[0].message, "content", None))
                print(response.choices[0].message.content)
                return [response.choices[0].message.content]
        else:
            return [choice.message.content for choice in response.choices]
