"""ChromaDB database for storing chunks and their embeddings"""

import chromadb

client = chromadb.PersistentClient(path="travel_database")
collection = client.get_or_create_collection(name="travel_guides")

def add_documents(documents, embeddings, metadata, ids):
    """Add documents to collection"""
    collection.upset(
        documents=documents,
        embeddings=embeddings,
        metadata=metadata,
        ids=ids
    )

def search_database(query_embedding, top_k=3):
    """Search database for chunks and their embeddings"""
    results = collection.query(query_embedding=[query_embedding], n_results=top_k)
    return results

def database_count():
    """Count database"""
    return collection.count()

if __name__ == "__main__":
    #testing
    print("Number of stored chunks: ", database_count())