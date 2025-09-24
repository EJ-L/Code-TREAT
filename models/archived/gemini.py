from APIBase import APIBase
from utils import *
import time
import google.generativeai as genai

# https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4
class GPT(APIBase):
    def __init__(self, api_key, base_url, model_name, client):
        model_names = ['gemini-1.0-pro', 'gemini-1.5-pro', 'gemini-1.5-flash']
        if model_name not in model_names:
            raise ValueError(f"Invalid model name. Please choose from {model_names}")
        super().__init__(api_key, base_url, model_name)
        self.setup_client(client)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def gemini_chat(
        self,
        model,                      
        messages,                   # [{'role': 'user', 'parts': "In one sentence, explain how a computer works to a young child."}, {'role': "model', 'parts': "A computer is like a very smart machine that can understand and follow our instructions, help us with our work, and even play games with us!"}
        temperature=temperature,    # [0, 2]: Lower values -> more focused and deterministic; Higher values -> more random.
        n=1,                        # Chat response choices to generate for each input message.
        max_tokens=1024,            # The maximum number of tokens to generate in the chat completion.
        delay=delay_time            # Seconds to sleep after each request.
    ):
        time.sleep(delay)
        response = self.client.generate_content(
            messages,
            generation_config=genai.types.GenerationConfig(
                # Only one candidate for now.
                candidate_count=n,
                # stop_sequences=['x'],
                max_output_tokens=max_tokens,
                temperature=temperature)
        )   
        
        if n == 1:
            return response.text
        else:
            return [candidate.text for candidate in response.candidates]