#%%
import os
import functools
import operator
from pprint import pprint
from typing_extensions import TypedDict
from typing import List, Sequence
from rag.search_tool import tavily_search
from rag.gen_index import IdexRetriever
from rag.grade_retrieval import GradeRetrieval
from rag.grade_hallucinations import GetHallucinations
from rag.grade_answer import GradeGenAnswer
from rag.generetor import Generetor
from rag.router import Router
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
#%%
# class to define the state graph
class GraphState(TypedDict):
    """ """

    query: str
    generation: str
    web_search: str
    documents: List[str]


class RagGraph:

    def __init__(self) -> None:
        pass
    
    
    def __retrive(self, state:dict, vts_type:str):

        # Get qury from the estate
        query = state['query']
        # Get document retrieved from the db
        retriever = IdexRetriever.get_index_retriever(vts_type=vts_type)
        # Documents
        documents = retriever.invoke(query)

        # Return and update the graph state
        return {"documents": documents, "query": query}
    
    def __grade_document(self, state:dict, model_type:str):
        
        # Get query and document retrieved 
        query = state['query']
        documents = state['documents']

        # Instance of scorer 
        retrieval_grade = GradeRetrieval.greade_document(model_type)

        # grade each document retrieved 
        filetere_doc = []
        # flag
        web_search = "No"
        #loop
        for doc in documents:
            score = retrieval_grade.invoke(
                {"query": query, "document": doc}
            )

            # Get grade
            grade:str = score['score']
            # get the documens are relevants
            if grade.casefold() == 'yes':
                filetere_doc.append(doc)

            else:
                web_search = 'Yes'
                continue
        
        # Update the graph state 
        return {
        "documents":filetere_doc, "query":query, 
        "web_search": web_search
        }
        
    def __web_search(self, state:dict):

        # get query and docuemnt from the state
        query = state['query']
        documents = state['documents']

        # invoke tool to make web searchs based on the query
        docs = tavily_search.invoke({"query":query})
        # format result
        web_result = "\n".join([doc['content'] for doc in docs])
        web_result = Document(page_content=web_result)
        if documents is not None:
            documents.append(web_result)
        else:
            documents = [web_result]

        # Update state
        return {
            "documents":documents,
            "query": query
        }
    
    def __generate(self, state:dict, model_type:str):

        # get the query and the docuemnts from the state
        query = state['query']
        documents = state['documents']

        # instace of generator
        generator = Generetor.generetor(model_type)

        # get results 
        gen_result = generator.invoke(
            {
                "context": documents,
                "query": query
            }
        )
        # Update state
        return {
            "documents":documents,
            "query": query,
            "generation": gen_result
        }
    
    def __route(self, state:dict, model_type:str):

        # Get query from the estate 
        query = state['query']
        # Define the route getting the datasource
        router = Router.query_router(model_type=model_type)
        # datasource
        source = router.invoke({'query':query})

        # route
        if source['datasource'] == 'web_search':
            return 'websearch'
        elif source['datasource'] == 'vectorstore':
            return 'vectorstore'
        
    def __decide_to_generate(self, state:dict):

        # Get query, web_search state and docuemnts 
        query = state['query']
        web_search = state['web_search']
        filtered_document = state['documents']
        
        # if the state of web_search is yes, decide re-generate to a new query from the a web search directly
        if web_search == 'Yes':
            return 'websearch'
        else:
            return 'generate'
        
    def __hallucinations_grade_generation(self, state:dict, model_type:str):

        # Get info from the state
        query = state['query']
        documents = state['documents']
        generation = state['generation']

        # GET hallucination and grade the answer
        get_hallucinations = GetHallucinations.get_hallucinations(model_type=model_type)
        grade_gen = GradeGenAnswer.greade_answer(model_type=model_type)
        # Get hallucination score
        score_h = get_hallucinations.invoke({'documents':documents, 'generation': generation})

        if score_h['score'].lower() == 'yes':
            # the answer is grounded by the context[documents]
            # Now get the score from the answer
            grade = grade_gen.invoke({'query': query, 'generation':generation})
            # the generation wheater can addresses the query
            if grade['score'].lower() == 'yes':
                return 'useful'
            else:
                return 'not useful'
            
        else:
            return 'not supported'
        
    def graph(self, user_query:str):

        # define node functions
        retriever_node = functools.partial(self.__retrive, vts_type='faiss')
        grade_documets_node = functools.partial(self.__grade_document, model_type='openai')
        # web_search_node = functools.partial(self.__web_search, model_type='openai')
        generetor_node = functools.partial(self.__generate, model_type='openai')
        # Conditional edges functions
        query_route = functools.partial(self.__route,  model_type='openai')
        hallucinations_grade_generation = functools.partial(self.__hallucinations_grade_generation,  model_type='openai')

        # graph
        workflow = StateGraph(GraphState)

        # Create nodes
        workflow.add_node('websearch', self.__web_search)
        workflow.add_node('retrieve', retriever_node)
        workflow.add_node('grade_documents', grade_documets_node)
        workflow.add_node('generate', generetor_node)

        # Create a conditional entrypoint
        workflow.set_conditional_entry_point(
            query_route,
            {
                'websearch':'websearch',
                'vectorstore':'retrieve',
            },
        )

        # create edges
        workflow.add_edge('retrieve', 'grade_documents')
        # add conditional edges
        workflow.add_conditional_edges(
            'grade_documents',
            self.__decide_to_generate,
            {
                'websearch':'websearch',
                'generate':'generate',
            }
        )

        workflow.add_edge('websearch', 'generate')

        workflow.add_conditional_edges(
            'generate',
            hallucinations_grade_generation,
            {
                'not supported':'generate',
                'useful':END,
                'not useful': 'websearch'
            }
        )

        rag_app = workflow.compile()

        # Reuslts
        for output in rag_app.stream(
            {
                'query':user_query
            }
        ):
            pprint(output)
            print('------')

        return output['generation']
#%%
# query = ''
#%%
# instance
# cls_instace = RagGraph()

# ouput = cls_instace.graph(user_query=query)


