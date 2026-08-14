# ChromaDB database for storing chunks and their embeddings

import chromadb


client = chromadb.PersistentClient(path="travel_database")
collection = client.get_or_create_collection(name="travel_guides")


def add_documents(documents, embeddings, metadata, ids, batch_size=5000):
    # Add document chunks to ChromaDB in smaller batches.
    total_documents = len(documents)

    for start in range(0, total_documents, batch_size):
        end = min(start + batch_size, total_documents)

        collection.upsert(
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadata[start:end],
            ids=ids[start:end],
        )

        print(
            f"Stored chunks {start + 1} through {end} "
            f"of {total_documents}."
        )


def search_database(query_embedding, top_k=3, where=None):
    # Search for similar records, optionally filtering by metadata.
    search_arguments = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }

    if where:
        search_arguments["where"] = where

    return collection.query(**search_arguments)


def database_count():
    return collection.count()
