from sentence_transformers import SentenceTransformer

# Same embedding model from class notes
# This model converts text into numerical vectors that represent meaning
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

"""
    Convert multiple chunks of travel-guide text into embeddings.

    Each chunk becomes a list of numbers that can later be stored
    and searched in the ChromaDB vector database.
    """

def create_embeddings(text_chunks):
    embeddings = embedding_model.encode(text_chunks)
    return embeddings.tolist()

"""
    Convert one user's travel question into an embedding.

    The database will compare this vector with the stored travel-guide
    vectors to find the most relevant information.
    """

def create_query_embeddings(question): # Converts user question into embedding
    embedding = embedding_model.encode([question])
    return embedding[0].tolist()

#This is a small test to make sure the embedding functions work
if __name__ == '__main__':
    test_chunks = [
        "Chicago has museums, restaurants, and music festivals.",
        "Orlando is known for theme parks and family attractions."
    ]

test_embedding = create_embeddings(test_chunks)

print("Number of embeddings: ", len(test_embedding))
print("Embedding size: ", len(test_embedding[0]))

question = "What can I do in Chicago?"
query_embedding = create_query_embeddings(question)

print("Question embedding size: ", len(query_embedding))