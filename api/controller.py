from fastapi import FastAPI, Query

try:
    from api import service
except ImportError:
    import service 

app = FastAPI()

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
