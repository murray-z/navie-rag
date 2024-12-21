from store_data import StoreData
from ollama import chat
from config import LLM_MODEL_NAME, RAG_CHAT_TEMPLATE, SCORE_THRESHOLD


# 初始化vector_store
vector_store = StoreData()


if __name__ == '__main__':
    query = "人工智能与人类智能的关系问题，从什么时候在国内外就进行了非常激烈的辩论。"
    res = vector_store.search(query, top_k=1)
    doc, score = res[0]
    if score > SCORE_THRESHOLD:
        context = doc.page_content
        source = doc.metadata["source"]
    else:
        context = ""

    user_message = RAG_CHAT_TEMPLATE.format(question=query, context=context)
    print("============USER MESSAGE===================")
    print(user_message)
    print("============================================\n\n")
    response = chat(model=LLM_MODEL_NAME, messages=[{'role': 'user', 'content': user_message}],
                    stream=True)
    print("============RESPONSE===================")
    for chunk in response:
        print(chunk['message']['content'], end='', flush=True)


