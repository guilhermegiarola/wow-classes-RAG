from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

try:
    from api import service
except ImportError:
    import service

app = FastAPI()

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React default dev server
        "http://localhost:5173",  # Vite default dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get('/')
def read_root():
    return {'message': 'WoW Classes RAG API - Running!'}

@app.get('/retrieve-answers')
def retrieve_answers(query_text: str = Query(..., description="The question to ask about WoW classes")):
    return service.retrieve_answers(query_text)

@app.get('/generate-knowledge-base')
def web_scrape_data():
    return service.generate_knowledge_base()

@app.get('/generate-embedding-vector')
def generate_embedding_vector():
    return service.generate_vectorized_knowledge_base()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
