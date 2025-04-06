from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from retrievers.bm25_retriever import PABM25Retriever

app = FastAPI()

documents_store: Dict[str, Document] = {}
retriever = PABM25Retriever.from_documents([])


class DocumentInput(BaseModel):
    """Model for adding a single document."""

    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    id: Optional[str] = None


class DocumentsInput(BaseModel):
    """Model for adding multiple documents."""

    documents: List[DocumentInput]


class SearchResponseItem(BaseModel):
    """Single search result item."""

    id: Optional[str]
    page_content: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Model for search response."""

    results: List[SearchResponseItem]


def rebuild_retriever():
    """
    Rebuild the BM25 retriever based on the current documents store.
    This function should be called whenever documents are added or removed.
    """
    global retriever
    all_docs = list(documents_store.values())
    retriever = PABM25Retriever.from_documents(all_docs)


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/bm25/documents")
async def add_documents(doc_list: DocumentsInput):
    """
    Add multiple documents to the BM25 index. If an id is not provided,
    one will be generated automatically.

    Args:
        doc_list (DocumentListInput): A list of documents to add.
    """
    for doc_input in doc_list.documents:
        doc_id = doc_input.id or f"doc_{len(documents_store) + 1}"
        if doc_id in documents_store:
            raise HTTPException(
                status_code=400, detail=f"Document with id {doc_id} already exists."
            )
        new_doc = Document(
            page_content=doc_input.page_content, metadata=doc_input.metadata, id=doc_id
        )
        documents_store[doc_id] = new_doc

    # Rebuild the retriever with updated docs
    rebuild_retriever()
    return {"message": "Documents added successfully."}


@app.delete("/api/v1/bm25/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a document by its ID.

    Args:
        doc_id (str): The ID of the document to remove.
    """
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found.")
    del documents_store[doc_id]

    # Rebuild the retriever after deletion
    rebuild_retriever()

    return {"message": f"Document {doc_id} deleted successfully."}


@app.get("/api/v1/bm25/search", response_model=SearchResponse)
async def search_documents(query: str, k: Optional[int] = None):
    """
    Search for documents using the BM25 retriever.

    Args:
        query (str): The query string.
        k (int, optional): Number of documents to return. Defaults to retriever's k.
    """
    # If k is provided, temporarily adjust retriever's k, otherwise use the default.
    original_k = retriever.k
    if k is not None:
        retriever.k = k

    try:
        results = retriever._get_relevant_documents(query)
    finally:
        # Restore original k
        retriever.k = original_k

    response_items = [
        SearchResponseItem(
            id=doc.id, page_content=doc.page_content, metadata=doc.metadata
        )
        for doc in results
    ]
    return SearchResponse(results=response_items)


@app.put("/api/v1/bm25/documents/{doc_id}")
async def update_document(doc_id: str, doc_input: DocumentInput):
    """
    Update an existing document by its ID.

    Args:
        doc_id (str): The ID of the document to update.
        doc_input (DocumentInput): The updated document data.

    Raises:
        HTTPException: If the document doesn't exist.
    """
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found.")

    updated_doc = Document(
        page_content=doc_input.page_content, metadata=doc_input.metadata, id=doc_id
    )
    documents_store[doc_id] = updated_doc

    # Rebuild the retriever to reflect the updated document
    rebuild_retriever()

    return {"message": f"Document {doc_id} updated successfully."}
