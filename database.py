"""
Store and search travel-guide chunks with a ChromaDB vector database.

This file is the database layer. Other files call these small functions instead
of needing to know all of ChromaDB's commands. ChromaDB stores three connected
things for each record: the original text, its numerical embedding, and useful
metadata such as the city or source filename.

Functions from class notes:
    - add_documents (adapted from the class collection.add example)
    - search_database (adapted from the class collection.query example)
    - database_count (adapted from the class collection.count example)

Functions I did:
    - get_all_documents

I also added persistent storage, batching, and upsert behavior to the class
database pattern so it could handle the much larger travel dataset.
"""

# From class notes: import ChromaDB.
import chromadb


# My change: use a persistent client so the travel database survives restarts.
client = chromadb.PersistentClient(path="travel_database")

# From class notes, adapted: create or reopen the collection.
collection = client.get_or_create_collection(name="travel_guides")


def add_documents(documents, embeddings, metadata, ids, batch_size=5000):
    """
    Add or update document chunks in ChromaDB in manageable batches.

    ChromaDB limits how much information can be sent in one request. Breaking a
    large list into batches prevents an oversized-request error. ``upsert``
    means "insert this ID if it is new, or update it if it already exists."

    The document, embedding, metadata, and ID lists must stay in the same order
    because matching positions describe the same stored chunk.
    """

    total_documents = len(documents)

    # My change: store the large dataset in batches instead of one giant request.
    for start in range(0, total_documents, batch_size):
        end = min(start + batch_size, total_documents)

        # Upsert adds new IDs and updates IDs that are already stored.
        collection.upsert(
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadata[start:end],
            ids=ids[start:end],
        )

        # Print progress because this can take a while with the full dataset.
        print(
            f"Stored chunks {start + 1} through {end} "
            f"of {total_documents}."
        )


def search_database(query_embedding, top_k=3):
    """
    Return the ChromaDB chunks closest to a question embedding.

    The result includes the text, metadata, and distance for each match. A
    smaller distance means the chunk is more closely related to the question.
    """

    # From class notes: query ChromaDB for the top matching chunks.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return results


def database_count():
    """
    Return the number of records currently stored in the collection.

    The RAG test uses this to decide whether it needs to build the knowledge
    base before trying a search.

    A count of zero means the knowledge base has not been built yet.
    """

    return collection.count()


def get_all_documents(batch_size=5000):
    """
    Return every stored chunk and its metadata from ChromaDB.

    This is used for geographic searches. ChromaDB can compare embeddings, but
    it cannot calculate miles or kilometers between latitude/longitude points.
    The RAG code gets all records and performs that distance math in Python.

    Embeddings are left out because geographic search only needs document text
    and metadata, which saves some memory.
    """

    documents = []
    metadatas = []

    # My change: read large collections in smaller groups. ChromaDB uses an
    # internal database that cannot return tens of thousands of IDs at once.
    for offset in range(0, collection.count(), batch_size):
        batch = collection.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        documents.extend(batch["documents"])
        metadatas.extend(batch["metadatas"])

    return {
        "documents": documents,
        "metadatas": metadatas,
    }
