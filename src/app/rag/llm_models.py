import os
from dotenv import load_dotenv
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_community.chat_models.ollama import ChatOllama

load_dotenv()

openai_llm = AzureChatOpenAI(
    azure_deployment=os.getenv('AZURE_MODEL'),
    azure_endpoint= os.getenv('AZURE_OPENAI_API_BASE'),
    openai_api_version = '2023-07-01-preview',
    openai_api_type='azure',
    temperature= 0.001
    )

llama_llm_json =  ChatOllama(
    model='llama3:8b',
    num_thread= 8,
    format = 'json',
    temperature= 0.001
)

llama_llm =  ChatOllama(
    model='llama3:8b',
    num_thread= 8,
    temperature= 0.001
)