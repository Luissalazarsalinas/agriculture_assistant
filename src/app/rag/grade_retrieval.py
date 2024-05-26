#%%
import rag.llm_models as llm_models
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
#%%
class GradeRetrieval:


    def __init__(self) -> None:
        pass

    @staticmethod
    def greade_document(model_type:str):

        prompt_llama = ChatPromptTemplate.from_template(
             """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
            Eres un evaluador senior, especializado en evaluar la reelevancia de un documento(estos documentos pueden estar en idioma INGLES) recuperado para dar respuesta a la consulta de un usuario.
            Si el documento contiene palabras claves relacionadas con la consulta del usuario, evalualo como reelevante.
            No es necesario que sea una prueba estricta. El objectivo es filtrar de manera efectiva, informacion recuperada que sea poco reelevante o erronea para dar respuesta a la consulta del usuario.
            En este sentido, Debes dar un puntaje binario 'yes' or 'no', con el fin de indicar si el documento recuperado es reelevante o no para respoder la consulta del usuario.
            Tu resultado lo debes entragar como un puntaje binario en formato JSON con una sola clave 'score' y sin preambulos, comentarios ni explicaciones.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Aqui tienes el docuemento recuperado: {document}
            Aqui tienes la consulta del usuario: {query}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """

        )

        prompt_openai = ChatPromptTemplate.from_template(
            """
            Eres un evaluador senior, especializado en evaluar la reelevancia de un documento(estos documentos pueden estar en idioma INGLES) recuperado para dar respuesta a la consulta de un usuario.
            Si el documento contiene palabras claves relacionadas con la consulta del usuario, evalualo como reelevante.
            No es necesario que sea una prueba estricta. El objectivo es filtrar de manera efectiva, informacion recuperada que sea poco reelevante o erronea para dar respuesta a la consulta del usuario.
            En este sentido, Debes dar un puntaje binario 'yes' or 'no', con el fin de indicar si el documento recuperado es reelevante o no para respoder la consulta del usuario.
            Tu resultado lo debes entragar como un puntaje binario en formato JSON con una sola clave 'score' y sin preambulos, comentarios ni explicaciones.
            
            Aqui econtraras el documento recuperado: <documento>{document}</documento>
            Aqui encotraras la consulta del usuario: <query>{query}</query>
            """

        )

        # chain 
        if model_type == 'openai':

            #chain 
            chain_openai = prompt_openai | llm_models.openai_llm | JsonOutputParser()
            
            return chain_openai
        else:

            #chain
            chain_llama = prompt_llama | llm_models.llama_llm_json | JsonOutputParser()

            return chain_llama

#%%
# Test
# doc = 'La agricultura sostenible: clave en el bienestar de las sociedades  \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nCambiar idioma del sitio\n\n \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nAlcaldía de Medellín\nPortal de la ciudad \n\n\n\n\n            Alcaldía de Medellín\n            Secretarias y Dependencias'
#%%
# grader = GradeRetrieval.greade_document(model_type='openai')

# grader.invoke({'query':'cultivos sostenibles', 'document':doc})