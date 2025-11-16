from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
import os


class LLM_Agent:
    def __init__(self, llm_type: str, model_name: str, temperature: float = 0.2):
        '''
        Initialize the LLM handler.
        Args: llm_type (str): The type of LLM. The options are: "openai", "anthropic", "google", "ollama", "mistral".
            model_name (str): The name of the LLM model.
            temperature (float): The temperature for the LLM.
        '''
    
        self.llm_type = llm_type
        self.model_name = model_name
        self.temperature = temperature
        # self.api_key = self.get_api_key()
        self.model = self.define_model()

    def define_model(self) -> BaseModel:
        '''
        Define the LLM model.
        output: BaseModel: The defined LLM model.
        '''
        if self.llm_type == "openai":
            return ChatOpenAI( model_name=self.model_name, temperature=self.temperature)
        elif self.llm_type == "anthropic":
            return ChatAnthropic(model=self.model_name, temperature=self.temperature)
        elif self.llm_type == "google":
            return ChatGoogleGenerativeAI(model=self.model_name, temperature=self.temperature)
        elif self.llm_type == "mistral":
            return ChatMistralAI(model=self.model_name, temperature=self.temperature)
        elif self.llm_type == "qwen" or self.llm_type == "deepseek":
            return ChatOllama(model=self.model_name, temperature=self.temperature)
        else:
            raise ValueError(f"Unknown LLM type: {self.llm_type}")

    def generate_prompt(self, args: dict, sys_prompt: str, user_prompt: str) -> list:
        '''
        Generate the prompt for the LLM.
        Args: args (dict): The arguments to be included in the prompt.
            sys_prompt (str): The system prompt.
            user_prompt (str): The user prompt.
        output: list: The generated prompt for the LLM.
        '''
        user_prompt = user_prompt.format(**args)
        message = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt)
        ]
        return message
    def invoke(self, prompt: list) -> str:
        '''
        Invoke the LLM with the given prompt.
        Args: prompt (list): The prompt for the LLM.
        output: str: The response from the LLM.
        '''
        return self.model.invoke(prompt).content
    
    
if __name__ == '__main__':
    llm = LLM_Agent(llm_type="mistral", model_name="mistral-large-latest")
    print(llm.invoke(["hi"]))