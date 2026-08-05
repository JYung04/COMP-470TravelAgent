# Backend for the travel-guide RAG chatbot
#
# This file wires together retrieval (rag.py) and generation (Llama 3.2:3b via Ollama).
# It builds a prompt from the retrieved chunks, sends it to the local model,
# and returns the answer along with the sources used.

import ollama
import Init

from database import database_count
from rag import retrieve_context

def ensure_database_ready():

    if database_count() > 0:
        return True

    Init.setup()

    return database_count() > 0



MODEL_NAME = "llama3.2:3b"

# Build a prompt that gives the model the retrieved chunks as context
def build_prompt(question, documents, metadatas):

    context_pieces = []

    for text, meta in zip(documents, metadatas):
        source_note = f"[{meta['city']} - {meta['source']}, page {meta['page_number']}]"
        context_pieces.append(f"{source_note}\n{text}")

    context_block = "\n\n".join(context_pieces)

    prompt = f"""You are a helpful travel guide assistant. Use ONLY the context below to answer the question.
If the context doesn't contain the answer, say you don't have that information.

Context:
{context_block}

Question: {question}

Answer:"""

    return prompt


# Send the prompt to the local Llama 3.2:3b model and return its response
def generate_answer(question, documents, metadatas):

    prompt = build_prompt(question, documents, metadatas)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


# Full RAG pipeline: retrieve relevant chunks, then generate an answer
# This is the main function the frontend should call
def answer_question(question, number_of_results=3):

    documents, metadatas, distances = retrieve_context(
        question,
        number_of_results=number_of_results
    )

    if not documents:
        return "I don't have any information on that.", [], [], []

    answer = generate_answer(question, documents, metadatas)

    return answer, documents, metadatas, distances


# for testing the pipeline without a frontend
if __name__ == "__main__":

    print("Collection size:", database_count())

    if not ensure_database_ready():
        print("Failed to setup database")
        raise SystemExit(1)

    print("Travel Guide Chatbot (type 'quit' to exit)")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in ("quit", "exit"):
            break

        if not question:
            continue

        answer, documents, metadatas, distances = answer_question(question)

        print("\nAssistant:", answer)

        print("\nSources:")
        for meta, dist in zip(metadatas, distances):
            print(
                f"  - {meta['city']} / {meta['source']} "
                f"(page {meta['page_number']}, distance {dist:.3f})"
            )


def generate_response(user_input, history=None):
    """
    Adapter function so app.py's expected interface matches
    what answer_question() actually returns.
    """
    answer, documents, metadatas, distances = answer_question(user_input)

    sources = [
        f"{meta['city']} - {meta['source']} (page {meta['page_number']})"
        for meta in metadatas
    ]

    return {
        "answer": answer,
        "sources": sources
    }
