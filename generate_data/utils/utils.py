import openai
import os

API_NAME_DICT = {
    'gpt3': ["text-davinci-003", "text-davinci-002", "code-davinci-002"],
    'chatgpt': ["gpt-3.5-turbo", "gpt-3.5-turbo-0301"]
}

def set_openai_key(key):
    os.environ["OPENAI_API_KEY"] = key

def get_openai_key():
    return os.getenv("OPENAI_API_KEY")

def set_proxy(proxy):
    os.environ["https_proxy"] = proxy
    

class BasePrompt:
    """Base class for all prompts."""

    def __init__(self):
        self.prompt = None
        self.response = None

    def build_prompt(self, prompt):
        self.prompt = prompt
        return self.prompt
    
    def get_openai_result(self, 
                          engine="text-davinci-003", 
                          temperature=0, 
                          max_tokens=1024, 
                          top_p=1.0, 
                          frequency_penalty=0.0, 
                          presence_penalty=0.0
                          ):
        openai.api_key = get_openai_key()

        if engine in API_NAME_DICT["gpt3"]:
            response = openai.Completion.create(
                model = engine,
                prompt = self.prompt,
                temperature = temperature,
                max_tokens = max_tokens,
                top_p = top_p,
                frequency_penalty = frequency_penalty,
                presence_penalty = presence_penalty,
            )
        elif engine in API_NAME_DICT["chatgpt"]:
            response = openai.ChatCompletion.create(
                model = engine,
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.prompt}
                ]    
            )
        else:
            print("[ERROR] Engine {engine} not found!".format(engine=engine))
            print("Available engines are as follows:")
            print(API_NAME_DICT)
            response = None

        if engine in API_NAME_DICT["chatgpt"]:
            self.response = response["choices"][0]["message"]["content"]
        else:
            self.response = response['choices'][0]['text']

        return self.response
    
    def parse_response(self):
        raise NotImplementedError
    
    def set_system_prompt(self, sys_prompt):
        self.system_prompt = sys_prompt