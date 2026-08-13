"""
This file manages the RAG pipeline for the travel-guide.

It loads travel guide PDFs and text files, splits their text into chunks,
creates embeddings, stores the chunks in ChromaDB, and retrieves
the most relevant info
"""

from pathlib import Path

from pypdf import PdfReader

from embed import create_embeddings, create_query_embeddings
from database import add_documents, search_database, database_count


# Main project folder
project_folder = Path(__file__).resolve().parent


# Folders that store the relevant data
city_folders = [
    "Boston",
    "Chicago",
    "NYC",
    "Orlando",
    "Washington DC"
]

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

        # Load PDF files
        for pdf_path in city_folder.glob("*.pdf"):
            print("Reading PDF:", city, pdf_path.name)

            try:
                reader = PdfReader(str(pdf_path))

                for page_number, page in enumerate(
                    reader.pages,
                    start=1
                ):
                    text = page.extract_text() or ""

                    if text.strip():
                        documents.append(
                            {
                                "text": text,
                                "city": city,
                                "source": pdf_path.name,
                                "page_number": page_number,
                                "source_type": "PDF"
                            }
                        )

            except Exception as error:
                print("Could not read:", pdf_path.name)
                print(error)

        # Load TXT files, including files in subfolders
        # These files come from a script I ran extracting data from openstreetmap/ overpass
        for text_path in city_folder.glob("**/*.txt"):
            print("Reading TXT:", city, text_path.name)

            try:
                text = text_path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )

                if text.strip():
                    documents.append(
                        {
                            "text": text,
                            "city": city,
                            "source": text_path.name,
                            "page_number": 1,
                            "source_type": "TXT"
                        }
                    )

            except Exception as error:
                print("Could not read:", text_path.name)
                print(error)

    return documents

# Split loaded documents into chunks and create metadata for every chunk
def create_document_chunks(documents):

    document_chunks = []
    document_metadata = []

    for document in documents:
        chunks = split_text_into_chunks(
            document["text"],
            chunk_size=150,
            overlap=30
        )

        for chunk_number, chunk in enumerate(chunks):
            document_chunks.append(chunk)

            document_metadata.append(
                {
                    "city": document["city"],
                    "source": document["source"],
                    "page_number": document["page_number"],
                    "chunk_number": chunk_number,
                    "source_type": document["source_type"]
                }
            )

    return document_chunks, document_metadata

# Load documents, create chunks and embeddings, and store them in ChromaDB
def build_knowledge_base():

    documents = load_documents()

    document_chunks, document_metadata = create_document_chunks(
        documents
    )

    if not document_chunks:
        print("No readable document text was found.")
        return

    print("Creating embeddings...")

    chunk_embeddings = create_embeddings(
        document_chunks
    )

    chunk_ids = [
        f"travel_chunk_{index}"
        for index in range(len(document_chunks))
    ]

    add_documents(
        documents=document_chunks,
        embeddings=chunk_embeddings,
        metadata=document_metadata,
        ids=chunk_ids
    )

    print(
        len(chunk_embeddings),
        "chunk embeddings created."
    )

    print(
        "Collection size:",
        database_count()
    )

# Retrieve the Top-3 most relevant travel-guide chunks
def retrieve_context(question, number_of_results=3, city=None):

    question_embedding = create_query_embeddings(
        question
    )

    results = search_database(
        question_embedding,
        top_k=number_of_results,
        city=city
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return documents, metadatas, distances
