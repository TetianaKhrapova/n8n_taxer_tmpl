import chromadb

chroma_client = chromadb.HttpClient(host='localhost', port=5000)

# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(name="my_collection4")

# switch `add` to `upsert` to avoid adding the same documents every time
collection.upsert(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids=["id1", "id2"]
)

results = collection.query(
    query_texts=["This is a query document about florida"], # Chroma will embed this for you
    n_results=2 # how many results to return
)

print(results)


collections = chroma_client.list_collections()
print(collections)
collection_names = [col.name for col in collections]
print("Available collections :", collection_names)
