from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import rag_route

#app variable
app = FastAPI(
    title="AI AGRICULTURE ASSISTAN",
    description="An AI agriculture assistan",
    version= "0.0.1"
)

# add cors origins and middleware
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routes
app.include_router(rag_route.router)

# if __name__ == '__main__':
#     import uvicorn

#     uvicorn.run(app)