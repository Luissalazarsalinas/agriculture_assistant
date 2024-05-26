from pydantic import BaseModel, Field

# Rag inputs
class RagInput(BaseModel):

    query:str