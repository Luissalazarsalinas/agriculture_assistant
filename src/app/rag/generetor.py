#%%
import rag.llm_models as llm_models
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
#%%
class Generetor:


    def __init__(self) -> None:
        pass

    @staticmethod
    def generetor(model_type:str):

        prompt_llama = ChatPromptTemplate.from_template(
            """ 
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Eres un asistente senior especialista en tareas de preguntas y respuestas.
            Tus tareas son las siguientes:
            1. Debes responder la consulta de un usuario utilizando un contexto recuperado de informacion externa.
               1.1 No te limites solo al contexto, apoyate tambien en tu conocimiento.
               1.2 NO intentes adivinar la respuesta, si no sabes o no tienes conocimiento, debes responder lo siguiente: No cuento con conocimiento para responder tu cosulta.
            2. Elabora muy bien tu respuesta, la cual debe estar muy bien redactada y con calidad de respuesta HUMANA.
               2.1 No incluyas comentarios ni explicaciones en tu respuesta.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            consulta: {query}
            contexto: {context}
            respuesta: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        )

        prompt_openai = ChatPromptTemplate.from_template(
            """ 
            Eres un Ingeniero agronomo senior, altamente tecnico especialista en tareas de preguntas y respuestas.
            Tus tareas son las siguientes:
            1. Debes responder la consulta de un usuario utilizando un contexto recuperado de informacion externa.
               1.1 No te limites solo al contexto, apoyate tambien en tu conocimiento.
               1.2 NO intentes adivinar la respuesta, si no sabes o no tienes conocimiento, debes responder lo siguiente: No cuento con conocimiento para responder tu cosulta.
            2. Elabora muy bien tu respuesta, la cual debe estar muy bien redactada y con calidad de respuesta HUMANA.
               2.1 No incluyas comentarios ni explicaciones en tu respuesta.
            
            Aqui encontraras la consulta: {query}
            Aqui encontraras el contexto: {context}
            """

        )

        # chain 
        if model_type == 'openai':

            #chain 
            chain_openai = prompt_openai | llm_models.openai_llm | StrOutputParser()
            
            return chain_openai
        else:

            #chain
            chain_llama = prompt_llama | llm_models.llama_llm | StrOutputParser()

            return chain_llama
#%%
# docs = ["Document(page_content='La definición de agricultura sostenible, en detalle, no es algo inmutable. De acuerdo con las políticas agrarias de la Unión Europea, los factores que influyen en la sostenibilidad de los cultivos son:', metadata={'source': 'https://www.bbva.com/es/sostenibilidad/la-agricultura-sostenible-herramienta-clave-contra-el-hambre-y-el-cambio-climatico/', 'title': 'La agricultura sostenible: herramienta clave contra el cambio climático', 'description': 'La apuesta por la agricultura sostenible, aquella que es respetuosa con el medioambiente, rentable y social, es en una prioridad.', 'language': 'es'})",
#  "Document(page_content='aportación de materia orgánica para la conservación de fertilidad del suelo; el uso de “pesticidas” naturales para combatir plagas y malas hierbas.  Agricultura Biodinámica La agricultura sostenible biodinámica se basa en la interacción entre el suelo, los nutrientes, los microorganismos, los animales y los cultivos, y las relaciones energéticas entre estos elementos. La palabra biodinámica tiene origen griego y significa “ciencia que estudia las fuerzas vitales de la vida”. Este modelo de agricultura sostenible nace de la teoría de antroposofía desarrollada por Rudolf Steiner a principios del siglo XX. Este modelo de agricultura sostenible se caracteriza por el uso de compuestos específicos de procedencia animal y vegetal preparados durante meses, y teniendo en cuenta los ciclos astronómicos para siembra, labranza, mantenimiento y cosecha.  Permacultura La permacultura, o agricultura sostenible permanente, surge en Australia al observar las pingües interrelaciones en el ecosistema de la selva. Su principal objetivo es ajustarse lo máximo posible a la naturaleza, igual a como lo hacían los pueblos indígenas durante siglos, asimismo se llegaría a la agricultura sostenible y más eficiente. Sus características principales son:  el estudio del terreno donde se sitúa el cultivo para determinar los organismos que lo habitan durante al año; la distribución de los cultivos para que dichas interrelaciones se lleven a cabo; la utilización multifuncional y sostenible de los elementos, aprovechando al máximo las', metadata={'source': 'https://eos.com/es/blog/agricultura-sostenible/', 'title': 'Agricultura Sostenible: La Aplicación Del Nuevo Concepto', 'description': 'El concepto de agricultura sostenible, los principios y las prácticas, los modelos y las ventajas. El ejemplo de la agricultura sostenible en Argentina.', 'language': 'es'})",
#  "Document(page_content='necesarias para conseguir la agricultura sostenible. Ventajas De La Agricultura Sostenible El argumento más común en contra de la agricultura sostenible es que no puede “alimentar al mundo” debido al control del suelo y a la gestión de cultivos. Pero vamos a considerar las ventajas de la estrategia agrícola sostenible. La influencia destructiva de la agricultura sostenible en el medioambiente es mínima, puesto que trata de usar las tecnologías y métodos menos dañinos. Las granjas sostenibles no usan pesticidas o fertilizantes químicos, semillas modificadas genéticamente y antibióticos para los animales, ni tampoco generan cantidades tóxicas de residuos, gracias a la gestión de residuos de los cultivos. Todos estos factores tienen una influencia positiva en la salud pública, produciendo alimentos más saludables y haciendo el proceso de cultivo más seguro para los agricultores. La agricultura sostenible también promueve y apoya el bienestar de los animales. Los granjeros crían a sus animales en condiciones cercanas a las naturales para disminuir el estrés, el dolor, las enfermedades y el sufrimiento del ganado. El aspecto social de la agricultura sostenible implica el replanteamiento de las granjas familiares y comunidades rurales. Combinada con otras estrategias la agricultura sostenible puede ayudar a aumentar el nivel de ocupación, educación, salud, asimismo cómo cubrir las necesidades culturales y espirituales.  Acerca del autor:     Prof. Dr. Petro Kogut', metadata={'source': 'https://eos.com/es/blog/agricultura-sostenible/', 'title': 'Agricultura Sostenible: La Aplicación Del Nuevo Concepto', 'description': 'El concepto de agricultura sostenible, los principios y las prácticas, los modelos y las ventajas. El ejemplo de la agricultura sostenible en Argentina.', 'language': 'es'})",
#  "Document(page_content='La agricultura sostenible: clave en el bienestar de las sociedades  \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nCambiar idioma del sitio\n\n \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nAlcaldía de Medellín\nPortal de la ciudad \n\n\n\n\n            Alcaldía de Medellín\n            Secretarias y Dependencias', metadata={'source': 'https://www.medellin.gov.co/es/sala-de-prensa/noticias/la-agricultura-sostenible-clave-en-el-bienestar-de-las-sociedades/#:~:text=La%20agricultura%20sostenible%2C%20a%20diferencia,vez%20mitiguen%20el%20cambio%20clim%C3%A1tico.', 'title': 'La agricultura sostenible: clave en el bienestar de las sociedades  ', 'description': 'La agricultura sostenible, a diferencia de la agricultura convencional que traspasa los límites, se caracteriza por la producción segura.', 'language': 'es'})"]
# #%%
# gen = Generetor.generetor(model_type='openai')

# gen.invoke({'context':docs, 'query': 'cultivos sostenibles'})