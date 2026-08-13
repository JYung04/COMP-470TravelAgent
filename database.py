# ChromaDB database for storing chunks and their embeddings

import chromadb

client = chromadb.PersistentClient(path="travel_database")
collection = client.get_or_create_collection(name="travel_guides")

def add_documents(documents, embeddings, metadata, ids, batch_size = 5000):

    # Add document chunks to ChromaDB in smaller batches
    # After I added text files, the initial batch maxxed out, so had to move to smaller ones

    total_documents = len(documents)

    for start in range(0, total_documents, batch_size):
        end = min(
            start + batch_size,
            total_documents
        )

        collection.upsert(
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadata[start:end],
            ids=ids[start:end]
        )

        print(
            f"Stored chunks {start + 1} through {end} "
            f"of {total_documents}."
        )

#Search database for chunks and their embeddings
def search_database(query_embedding, top_k=3, city=None):

    where_filter = {"city": city} if city else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter
    )
    return results

# Count database
def database_count():

    return collection.count()
