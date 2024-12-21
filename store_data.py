from langchain_milvus import Milvus
from custom_embedding import CustomEmbeddings
from config import MILVUS_HOST, MILVUS_COLLECTION_NAME
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredExcelLoader, UnstructuredMarkdownLoader, UnstructuredCSVLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, OVERLAP_SIZE
from langchain.schema import Document
import os
import chardet


class StoreData:
    def __init__(self, embedding_function=CustomEmbeddings,
                 collection_name=MILVUS_COLLECTION_NAME,
                 milvus_host=MILVUS_HOST,
                 chunk_size=CHUNK_SIZE,
                 overlap_size=OVERLAP_SIZE):

        self.vector_store = Milvus(embedding_function=embedding_function(),
                                   collection_name=collection_name,
                                   connection_args={"uri": milvus_host},
                                   auto_id=True)

        self.doc_split = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                        chunk_overlap=overlap_size)

    def parse_file(self, file_path):
        if file_path.endswith(".pdf"):
            return PyMuPDFLoader(file_path).load()
        elif file_path.endswith(".txt"):
            with open(file_path, 'rb') as f:
                encoding = chardet.detect(f.read())['encoding']
            return TextLoader(file_path, encoding=encoding).load()
        elif file_path.endswith(".docx"):
            return UnstructuredWordDocumentLoader(file_path).load()
        elif file_path.endswith(".xlsx"):
            return UnstructuredExcelLoader(file_path).load()
        elif file_path.endswith(".md"):
            return UnstructuredMarkdownLoader(file_path).load()
        elif file_path.endswith(".csv"):
            return UnstructuredCSVLoader(file_path).load()
        elif file_path.endswith(".ppt"):
            return UnstructuredPowerPointLoader(file_path).load()
        else:
            print(f"Unsupported file type: {file_path.split('.')[-1]}")

    def parse_docs(self, file_dir):
        documents = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                file_path = os.path.join(root, file)
                documents.extend(self.parse_file(file_path))
        return documents

    def split_documents(self, documents):
        return self.doc_split.split_documents(documents)

    def add_documents(self, documents):
        new_documents = []
        for document in documents:
            new_documents.append(Document(page_content=document.page_content,
                                          metadata={"source": document.metadata["source"]}))
        # print(new_documents)
        self.vector_store.add_documents(new_documents)

    def store_data(self, file_dir):
        """存储数据"""
        documents = self.parse_docs(file_dir)
        # print(documents)
        split_docs = self.split_documents(documents)
        self.add_documents(split_docs)

    def search(self, query, top_k=1):
        """搜素"""
        return self.vector_store.similarity_search_with_score(query, k=top_k)


if __name__ == '__main__':
    store_data = StoreData()
    # store_data.store_data("./datas")
    res = store_data.search("了解下人工智能与人类智能")
    for res, score in res:
        print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")




