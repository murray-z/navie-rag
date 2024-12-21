from langchain.embeddings.base import Embeddings
from pydantic import BaseModel
import ollama
from config import EMBEDDING_MODEL_NAME

class CustomEmbeddings(Embeddings, BaseModel):
    def embed_query(self, text: str) -> list[float]:
        response = ollama.embed(model=EMBEDDING_MODEL_NAME, input=text)
        return response["embeddings"][0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = ollama.embed(model=EMBEDDING_MODEL_NAME, input=texts)
        return response["embeddings"]


if __name__ == '__main__':
    embedding = CustomEmbeddings()
    print(embedding.embed_documents(["What is the capital of France?", "你是谁"]))
    print(embedding.embed_query("What is the capital of France?"))




