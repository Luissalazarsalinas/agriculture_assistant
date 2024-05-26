#%%
import rag.llm_models as llm_models
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
#%%
class Router:


    def __init__(self) -> None:
        pass

    @staticmethod
    def query_router(model_type:str):

        prompt_llama = ChatPromptTemplate.from_template(
                """
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                Eres un Asistente experto enrutando las consultas de un usuario a la fuente de informacion mas optima entre una vectorstore o web search.

                Tus tareas son las siguientes:
                1. Debes usar la vectorstore para preguntas que contengan los siguientes topicos:
                    - Cultivos sostenibles
                   - Comercializacion de cultivos agricolas
                   - Redimiento de cultivos agricolas [Estadísticas de rendimiento, técnicas de optimización.]
                   - Clima principalmente ne Colombia
                   - Suelo [nutrientes, calidad etc.] principalmente en Colombia
                   1.1 No es necesario que seas tan estricto con las palabras clave en la consulta que puedan estar relacionada a los topicos mencionados anteriormente.
                2. Si en la consuta no apunta a ninguno de los topicos mencionados o algo relacionado, usa el web search.
                3. Basado en al consulta debes dar la siguiente seleccion binaria: 'vectorstore' o 'web_search'.
                4. Siempre debes retornar tu respuesta en formato JSON con una sola clave 'datasource', sin ningun tipo de preambulo o explicaciones.

                Aqui puedes encontrar la consulta que debe ser enrutada: <query>{query}</query>
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """
                )

        prompt_openai = ChatPromptTemplate.from_template(
             """
                
                Eres un Asistente experto enrutando las consultas de un usuario a la fuente de informacion mas optima entre una vectorstore o web search.

                Tus tareas son las siguientes:
                1. Debes usar la vectorstore para preguntas que contengan los siguientes topicos:
                   - Cultivos sostenibles, agricultura sostenible
                   - Comercializacion de cultivos agricolas
                   - Clima principalmente ne Colombia
                   - Suelo [nutrientes, calidad etc.] principalmente en Colombia
                   - Manejo integrado de plagas
                   1.1 No es necesario que seas tan estricto con las palabras clave en la consulta que puedan estar relacionada a los topicos mencionados anteriormente.
                2. Si en la consuta no apunta a ninguno de los topicos mencionados o algo relacionado, usa el web search.
                3. Basado en al consulta debes dar la siguiente seleccion binaria: 'vectorstore' o 'web_search'.
                4. Siempre debes retornar tu respuesta en formato JSON con una sola clave 'datasource', sin ningun tipo de preambulo o explicaciones.

                Aqui puedes encontrar la consulta que debe ser enrutada: <query>{query}</query>
                
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
#%%
# query = 'cultivos sostenibles'

# #%%
# route = Router.query_router(model_type='openai')
# route.invoke({'query':query})