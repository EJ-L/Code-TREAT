from APIBase import APIBase
class CodeLLaMa_Ins(APIBase):
    def __init__(self, api_key, base_url, model_size, client_library, system_prompt="You are a helpful asssitant"):
        available_sizes = ["7b", "13b", "34b", "70b"]
        # check the whether the model size is valid
        if model_size not in available_sizes:
            raise ValueError(f"Model size must be one of {available_sizes}")
        # set the model name
        model_name = f"codellama/CodeLlama-{model_size}-Instruct-hf"
        # call the parent class constructor
        super().__init__(api_key, base_url, model_name)
        self.text_based_messages =[
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        # self.image_based_messages = None
        self.setup_client(client_library) 