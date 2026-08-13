# Backend for the travel-guide chatbot

import ollama
from rag import retrieve_context

MODEL_NAME = "llama3.2:3b"
MAX_HISTORY_MESSAGES = 6
SYSTEM_PROMPT = """You are a helpful travel guide assistant. You have two sources of information:
1. Retrieved context - factual travel-guide details such as restaurants, attractions, hours, and locations.
2. The conversation history - the answers you've already given the user.
When the user asks you to modify, or refer back to something already discussed
(for example "make day 2 more relaxed"), rely on the conversation history, 
and use the retrieved context for any new factual details you need.
If neither source has the answer, say you don't have that information instead of guessing."""

# retrieved chunks + the user's question
def build_prompt(question, documents, metadatas):
    """
    Helper function for building prompt. Zips source data
    :param question:
    :param documents:
    :param metadatas:
    :return:
    """
    context_pieces = []

    for text, meta in zip(documents, metadatas):
        source_note = f"[{meta['city']} - {meta['source']}, page {meta['page_number']}]"
        context_pieces.append(f"{source_note}\n{text}")

    context_block = "\n\n".join(context_pieces)

    prompt = f"""Context:
{context_block}

Question: {question}

Answer:"""

    return prompt


def sanitize_history(history):
    """
    Helper function for generating sanitized history.
    Takes history and removes old and useless info
    Working but Incomplete needs more work
    :param history:
    :return:
    """
    if not history:
        return []

    sanitized = []

    for message in history:
        role = message.get("role")
        content = message.get("content")

        if role in ("user", "assistant") and content:
            sanitized.append({"role": role, "content": content})

    return sanitized[-MAX_HISTORY_MESSAGES:]

# ollama pipeline
def generate_answer(question, documents, metadatas, history=None):
    """
    Ollama Pipeline for generating answers.
    Sends the question directly to the ollama client
    :param question:
    :param documents:
    :param metadatas:
    :param history:
    :return:
    """
    turn_prompt = build_prompt(question, documents, metadatas)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(sanitize_history(history))
    messages.append({"role": "user", "content": turn_prompt})

    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages
    )

    return response["message"]["content"]


def answer_question(question, number_of_results=3, history=None, city=None):
    """
    Pipeline for RAG integration returns answer, documents, metadatas, distances
    :param question:
    :param number_of_results:
    :param history:
    :param city:
    :return:
    """
    documents, metadatas, distances = retrieve_context(question, number_of_results=number_of_results, city=city) # ChromaDB top-k relevant chunks

    if not documents:
        return "I don't have any information on that.", [], [], []

    answer = generate_answer(question, documents, metadatas, history=history)

    return answer, documents, metadatas, distances

# adapter for app.py
def generate_response(user_input, history=None, city=None):
    """
    Adapter function so app.py's expected interface matches what answer_question() actually returns.
    :param user_input:
    :param history:
    :param city:
    :return:
    """
    answer, documents, metadatas, distances = answer_question(
        user_input,
        history=history,
        city=city
    )

    sources = [
        f"{meta['city']} - {meta['source']} (page {meta['page_number']})"
        for meta in metadatas
    ]

    return {
        "answer": answer,
        "sources": sources
    }