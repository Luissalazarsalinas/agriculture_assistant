from fastapi import APIRouter, status
from rag.rag_graph import RagGraph
from routes.schema import RagInput

# Define route
router = APIRouter(
    prefix="/rag",
    tags=["RAG CHATBOT"]
)

# POST ROUTE
@router.post('/', status_code=status.HTTP_200_OK)
async def chatbot(user_query:RagInput):
    # convert query into a dict
    query_dict = user_query.model_dump()
    query = query_dict['query']

    # Instane of the rag
    rag = RagGraph()

    response = rag.graph(user_query=query)

    return {
        'CHABOT_RESPONSE':response
    }