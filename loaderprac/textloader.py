from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    file_path="../data/fun_rag_test_corpus.txt",
    encoding="utf-8"
    )
docs = loader.load()

print(docs)