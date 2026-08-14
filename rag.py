"""
This file manages the RAG pipeline for the travel guide.

It loads travel guide PDFs and text files, creates embeddings, stores the
documents in ChromaDB, and retrieves the most relevant information.
"""

from pathlib import Path

from pypdf import PdfReader

from database import add_documents, database_count, search_database
from embed import create_embeddings, create_query_embeddings


# Main project folder
project_folder = Path(__file__).resolve().parent


# City folders stored directly beside rag.py
city_folders = [
    "Boston",
    "Chicago",
    "NYC",
    "Orlando",
    "Washington DC",
]


# Streamlit city names mapped to the folder/database city names
city_names = {
    "new york city": "NYC",
    "new york": "NYC",
    "nyc": "NYC",
    "washington dc": "Washington DC",
    "boston": "Boston",
    "chicago": "Chicago",
    "orlando": "Orlando",
}


# Streamlit interests converted into terms used by the data categories
interest_terms = {
    "food & dining": "restaurants cafes food dining",
    "museums & history": "museums historic history",
    "nightlife": "bars nightlife entertainment",
    "outdoors & nature": "parks attractions outdoors nature",
    "shopping": "shopping",
    "family-friendly": "attractions entertainment parks museums family",
    "art & culture": (
        "museums historic entertainment attractions art culture"
    ),
    "sports": (
        "entertainment attractions sports stadium arena baseball basketball "
        "football hockey soccer"
    ),
}


# Split text into overlapping word chunks
def split_text_into_chunks(text, chunk_size=150, overlap=30):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# Load readable text from every PDF and TXT file in the city folders
def load_documents():
    documents = []

    for city in city_folders:
        city_folder = project_folder / city

        if not city_folder.exists():
            print("Folder not found:", city)
            continue

        # Load PDF files stored directly inside a city folder
        for pdf_path in city_folder.glob("*.pdf"):
            print("Reading PDF:", city, pdf_path.name)

            try:
                reader = PdfReader(str(pdf_path))

                for page_number, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""

                    if text.strip():
                        documents.append(
                            {
                                "text": text,
                                "city": city,
                                "source": pdf_path.name,
                                "page_number": page_number,
                                "source_type": "PDF",
                            }
                        )

            except Exception as error:
                print("Could not read:", pdf_path.name)
                print(error)

        # Recursively load TXT files from the category subfolders
        for text_path in city_folder.glob("**/*.txt"):
            print("Reading TXT:", city, text_path.name)

            try:
                text = text_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

                if text.strip():
                    documents.append(
                        {
                            "text": text,
                            "city": city,
                            "source": text_path.name,
                            "page_number": 1,
                            "source_type": "TXT",
                        }
                    )

            except Exception as error:
                print("Could not read:", text_path.name)
                print(error)

    return documents


# Split PDF pages, but keep each individual TXT place whole
def create_document_chunks(documents):
    document_chunks = []
    document_metadata = []

    for document in documents:
        if document["source_type"] == "TXT":
            chunks = [document["text"]]
        else:
            chunks = split_text_into_chunks(
                document["text"],
                chunk_size=150,
                overlap=30,
            )

        for chunk_number, chunk in enumerate(chunks):
            document_chunks.append(chunk)

            document_metadata.append(
                {
                    "city": document["city"],
                    "source": document["source"],
                    "page_number": document["page_number"],
                    "chunk_number": chunk_number,
                    "source_type": document["source_type"],
                }
            )

    return document_chunks, document_metadata


# Load documents, create embeddings, and store them in ChromaDB
def build_knowledge_base():
    documents = load_documents()

    document_chunks, document_metadata = create_document_chunks(documents)

    if not document_chunks:
        print("No readable document text was found.")
        return

    print("Creating embeddings...")

    chunk_embeddings = create_embeddings(document_chunks)

    chunk_ids = [
        f"travel_chunk_{index}"
        for index in range(len(document_chunks))
    ]

    add_documents(
        documents=document_chunks,
        embeddings=chunk_embeddings,
        metadata=document_metadata,
        ids=chunk_ids,
    )

    print(len(chunk_embeddings), "chunk embeddings created.")
    print("Collection size:", database_count())


# Retrieve the most relevant travel records
def retrieve_context(question, number_of_results=3):
    question_lower = question.lower()

    # Normalize the Streamlit city name to the database city name
    selected_city = next(
        (
            database_city
            for displayed_city, database_city in city_names.items()
            if displayed_city in question_lower
        ),
        None,
    )

    # Add terms matching the data categories for selected interests
    normalized_terms = []

    for interest, terms in interest_terms.items():
        if interest in question_lower:
            normalized_terms.append(terms)

    search_question = question

    if normalized_terms:
        search_question += "\nRelevant categories: " + " ".join(
            normalized_terms
        )

    # The initial Streamlit request needs more than three records
    if "day-by-day itinerary" in question_lower:
        number_of_results = max(number_of_results, 12)

    question_embedding = create_query_embeddings(search_question)

    where = {"city": selected_city} if selected_city else None

    results = search_database(
        question_embedding,
        top_k=number_of_results,
        where=where,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return documents, metadatas, distances
