"""
Turn travel text and user questions into numerical embedding vectors.

Computers cannot directly compare the meaning of two sentences. An embedding
model changes each sentence into a list of numbers. Text with similar meaning
usually receives nearby vectors, which lets ChromaDB find useful travel-guide
chunks for a question even when the wording is not exactly the same.

Functions from class notes:
    - create_embeddings (adapted from the class chunk-embedding example)
    - create_query_embeddings (adapted from the class query-embedding example)

Functions I did:
    - None in this file. I organized the class examples into reusable functions
      for the travel project.
"""

# From class notes: import the SentenceTransformer model.
from sentence_transformers import SentenceTransformer


# From class notes: load the MiniLM embedding model. The documents and questions
# have to use the same model or their vectors will not be comparable.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(text_chunks):
    """Create one embedding vector for every travel-guide text chunk."""

    # From class notes: create an embedding for each chunk.
    embeddings = embedding_model.encode(text_chunks)

    # ChromaDB works with regular lists, so convert the NumPy result.
    return embeddings.tolist()


def create_query_embeddings(question):
    """Create one embedding vector from the user's travel question."""

    # From class notes: put the question in a list before encoding it.
    embedding = embedding_model.encode([question])

    # There is only one question, so return its first (and only) vector.
    return embedding[0].tolist()
