from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from embeddings.bge_embedding import PABgeEmbeddings

app = FastAPI()

bge = PABgeEmbeddings(
    model_name="/Users/kevintao/Desktop/working/models/BAAI/bge-large-zh-v1.5"
)


# Request model for embedding documents
class DocumentsRequest(BaseModel):
    texts: List[str]


# Response model for embedding documents
class DocumentsResponse(BaseModel):
    embeddings: List[List[float]]


# Request model for embedding a query
class QueryRequest(BaseModel):
    text: str


# Response model for embedding a query
class QueryResponse(BaseModel):
    embedding: List[float]


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/bge/embed/documents", response_model=DocumentsResponse)
async def embed_documents(request: DocumentsRequest):
    """
    Embed a list of documents into vector embeddings using the PABgeEmbeddings class.

    Args:
        request (DocumentsRequest): A request containing a list of texts to embed.

    Returns:
        DocumentsResponse: A response containing a list of vector embeddings.
    """
    embeddings = bge.embed_documents(request.texts)
    return DocumentsResponse(embeddings=embeddings)


@app.post("/api/v1/bge/embed/query", response_model=QueryResponse)
async def embed_query(request: QueryRequest):
    """
    Embed a single query text into a vector embedding using the PABgeEmbeddings class.

    Args:
        request (QueryRequest): A request containing a single query text.

    Returns:
        QueryResponse: A response containing the query's vector embedding.
    """
    embedding = bge.embed_query(request.text)
    return QueryResponse(embedding=embedding)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
