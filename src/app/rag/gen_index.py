#%%
from rag.urls_data import Urls
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, WikipediaLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
#%%

# Create class
class IdexRetriever:
    """"""

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_index_retriever(vts_type:str):
        
        # Get a list with info from mainly web pages
        docs = [WebBaseLoader(data).load() for data in Urls]
        # Unpacked the list of list and get a 
        docs_list = [items for sublist in docs for items in sublist]

        # split and tokenize the documents
        text_aplitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size = 500,
            chunk_overlap = 0
            )
        
        doct_split = text_aplitter.split_documents(
            documents=docs_list
        )

        if vts_type == 'faiss':

            # Store data into faiss vectorstore
            vectorstore_faiss = FAISS.from_documents(
                documents= doct_split,
                embedding= GPT4AllEmbeddings()
            )

            retriever = vectorstore_faiss.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 7, 'fetch_k': 50}

            )

            return retriever
        else:
            # Store data into the vector store
            vectorstore_chroma = Chroma.from_documents(
                documents=doct_split,
                collection_name='RAG-APP',
                embedding= GPT4AllEmbeddings()
            )

            retriever = vectorstore_chroma.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 7, 'fetch_k': 50}
            )

            return retriever
#%%
# index = IdexRetriever.get_index_retriever(vts_type='faiss')

# index.invoke('cultivos sostenibles')