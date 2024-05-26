import os 
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

tavily_search = TavilySearchResults(max_results=10, search_depth='advanced')
