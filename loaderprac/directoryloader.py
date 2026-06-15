from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
loader = DirectoryLoader(
    path ="../data",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
)
docs = loader.load()
print(docs)