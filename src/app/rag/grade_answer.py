#%%
import rag.llm_models as llm_models
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
#%%
class GradeGenAnswer:


    def __init__(self) -> None:
        pass

    @staticmethod
    def greade_answer(model_type:str):

        prompt_llama = ChatPromptTemplate.from_template(
            """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Eres un evaluador senior, especializado en evaluar si una respuesta dada es util e iformativa para resolver la consulta de un usuario.
            En este sentido, DEBES dar una puntuacion binaria de 'yes' o 'no', con el fin de indicar si la respuesta resuleve de manera satisfactoria la consulta de un usuario. 
            De esta forma, Debes proporcionar la puntuacion en un formato JSON con una unica clave 'score' sin ningun preambulo ni explicaciones.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Aqui puedes encontrar respuesta:<generator>{generator}</generator>
            Aqui puedes encontrar la consulta:<query>{query}</query>
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """

        )

        prompt_openai = ChatPromptTemplate.from_template(
            """
            Eres un evaluador senior, especializado en evaluar si una respuesta dada es util e iformativa para resolver la consulta de un usuario.
            En este sentido, DEBES dar una puntuacion binaria de 'yes' o 'no', con el fin de indicar si la respuesta resuleve de manera satisfactoria la consulta de un usuario. 
            De esta forma, Debes proporcionar la puntuacion en un formato JSON con una unica clave 'score' sin ningun preambulo ni explicaciones.
    
            Aqui puedes encontrar respuesta:<generation>{generation}</generation>
            Aqui puedes encontrar la consulta:<query>{query}</query>
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
# gen = 'La agricultura sostenible es un enfoque de la producción agrícola que busca equilibrar la necesidad de producir alimentos con la preservación del medio ambiente y el bienestar social. Este tipo de agricultura se caracteriza por el uso de prácticas que minimizan el impacto ambiental, como la aportación de materia orgánica para mantener la fertilidad del suelo y el uso de pesticidas naturales para controlar plagas y malas hierbas. Además, la agricultura sostenible promueve el bienestar animal y el replanteamiento de las granjas familiares y comunidades rurales, contribuyendo así al desarrollo social y económico de las áreas rurales.\n\nExisten diferentes modelos de agricultura sostenible, como la agricultura biodinámica y la permacultura. La agricultura biodinámica se basa en la interacción entre el suelo, los nutrientes, los microorganismos, los animales y los cultivos, y considera los ciclos astronómicos para las actividades agrícolas. Por otro lado, la permacultura busca ajustarse lo máximo posible a la naturaleza y utiliza un enfoque multifuncional y sostenible de los recursos disponibles.\n\nEn resumen, la agricultura sostenible no solo busca producir alimentos de manera eficiente, sino que también se enfoca en la conservación de los recursos naturales, la protección del medio ambiente y la mejora de la calidad de vida de las comunidades rurales.'

# #%%
# grade = GradeGenAnswer.greade_answer(model_type='openai')

# grade.invoke({'query':'cultivos sostenibles', 'generation':gen})