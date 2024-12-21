EMBEDDING_MODEL_NAME = "bge-large:latest"
LLM_MODEL_NAME = "qwen2.5:0.5b"
MILVUS_COLLECTION_NAME = "navie_rag"
MILVUS_HOST = "./milvus_db/navie_rag.db"
CHUNK_SIZE=500
OVERLAP_SIZE=100
SCORE_THRESHOLD = 0.3
RAG_CHAT_TEMPLATE = """你是一个人工智能助手，你的任务是根据参考信息，回答用户的问题。
如果参考信息为空或者答案不在参考信息中，请回答‘对不起，无法给出准确答案。’

## 参考信息
{context}

## 问题
{question}

你的答案：
"""