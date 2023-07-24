from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext, Document

def load_knowledge() -> list[Document]:
    # Load data from directory
    documents = SimpleDirectoryReader('knowledge').load_data()
    return documents

def create_index() -> GPTVectorStoreIndex:
    print('Creating new index')
    # Load data
    documents = load_knowledge()
    # Create index from documents
    service_context = ServiceContext.from_defaults(chunk_size_limit=3000)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    # save_index(index)
    return index

def load_index() -> GPTVectorStoreIndex:
    # Load index from file
    try:
        index = GPTVectorStoreIndex.load_from_disk('knowledge/index.json')
    except FileNotFoundError:
        index = create_index()
    return index


def query_index(index: GPTVectorStoreIndex, prompt):
    # Query index
    query_engine = index.as_query_engine()
    response = query_engine.query(prompt)
    print(response)
    return response