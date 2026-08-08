"""
rag.py loads the travel data, splits it into manageable chunks, creates
embeddings, and stores them in ChromaDB. It also handles semantic searches and
location-based searches when the user asks about nearby places.

The basic path through this file is:

    travel files -> text chunks -> embeddings -> ChromaDB -> search results

Functions from class notes:
    - split_text_into_chunks
    - create_document_chunks (adapted from the class chunking examples)
    - build_knowledge_base (adapted from the class RAG setup steps)
    - retrieve_context

Functions I did:
    - parse_place_fields
    - parse_coordinates
    - load_documents
    - haversine
    - find_places_near
    - resolve_search_center
    - find_places_from_question
    - test_knowledge_base

The class notebook supplied the basic RAG structure. I changed it from campus
examples into a TXT-based tourism project and added the location search.
"""

# Used for the distance calculations in the haversine function.
import json
import difflib
import math
import re
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# Converts document text and user questions into embedding vectors.
from embed import create_embeddings, create_query_embeddings

# Database helpers for storing and retrieving the embedded travel data.
from database import (
    add_documents,
    database_count,
    get_all_documents,
    search_database,
)

project_folder = Path(__file__).resolve().parent

# Main folder containing the tourism files for each city.
data_folder = project_folder / "tourism_data"

# Only these city folders are included when the knowledge base is built.
city_folders = [
    "Boston",
    "Chicago",
    "NYC",
    "Orlando",
    "Washington DC",
]

# Cache neighborhood lookups so the free geocoder is not asked for the same
# place repeatedly. Please be nice to the free servers!
geocode_cache_file = project_folder / "geocode_cache.json"
last_geocode_request = 0.0

supported_city_names = {
    "boston": "Boston",
    "chicago": "Chicago",
    "new york": "NYC",
    "orlando": "Orlando",
    "washington": "Washington DC",
    "district of columbia": "Washington DC",
}

# This lets the caller use normal singular words such as "museum" or "bar."
category_aliases = {
    "attraction": "attractions",
    "bar": "bars",
    "cafe": "cafes",
    "historic site": "historic",
    "landmark": "historic",
    "museum": "museums",
    "park": "parks",
    "restaurant": "restaurants",
    "shop": "shopping",
}

question_categories = {
    "attractions": ["attractions"],
    "bars": ["bars"],
    "cafes": ["cafes"],
    "fun things": ["attractions", "entertainment", "parks", "museums", "nightlife"],
    "historic sites": ["historic"],
    "landmarks": ["historic", "attractions"],
    "museums": ["museums"],
    "nightlife": ["nightlife", "bars"],
    "parks": ["parks"],
    "place to eat": ["restaurants"],
    "places to eat": ["restaurants"],
    "restaurants": ["restaurants"],
    "shopping": ["shopping"],
}


def split_text_into_chunks(text, chunk_size=150, overlap=30):
    """Split a long string into smaller chunks with some overlap.

    Smaller chunks make retrieval more specific. The overlap keeps some context
    when an important sentence falls near the boundary between two chunks.
    """

    # My safety check: bad sizes could stop the loop from moving forward.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    # The overlap has to stay smaller than the chunk so the loop moves forward.
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least 0 and smaller than chunk_size")

    # From class notes: split the text by words and build overlapping chunks.
    words = text.split()

    chunks = []

    # Index of the first word in the next chunk.
    start = 0

    while start < len(words):
        # Slicing also handles the final chunk when fewer than 150 words remain.
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk:
            chunks.append(chunk)

        # Move forward while repeating the requested amount of context.
        start += chunk_size - overlap

    return chunks


def parse_place_fields(text):
    """
    Convert the ``Key: Value`` lines in a place file into a dictionary.
    """

    fields = {}

    for line in text.splitlines():
        line = line.strip()

        # Ignore blank lines and anything that does not match the file format.
        if not line or ":" not in line:
            continue

        # Split only once because values such as opening hours may contain colons.
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()

    return fields

#------------------------------------------------------------
# THIS IS FOR GEOLOCATION! USING HAVERSINE FORMULA BASICALLY!
#------------------------------------------------------------
def parse_coordinates(fields):
    """
    Converts latitude and longitude values into decimal numbers.

    """

    raw_latitude = fields.get("Latitude")
    raw_longitude = fields.get("Longitude")

    try:
        # Coordinates are stored as text, so they must be converted before use.
        latitude = float(raw_latitude)
        longitude = float(raw_longitude)
        return latitude, longitude

    # Missing values and placeholders such as "Not listed" are not usable.
    except (TypeError, ValueError):
        return None, None


def load_documents():
    """
    Read the TXT place files from each configured city folder.
    """

    documents = []

    for city in city_folders:
        city_folder = data_folder / city

        # Search category subfolders as well as the city folder itself.
        for text_path in city_folder.glob("**/*.txt"):
            print("Reading TXT:", city, text_path.name)

            try:
                # Skip a damaged character rather than losing the entire file.
                text = text_path.read_text(encoding="utf-8", errors="ignore")

                if text.strip():
                    # Parse fields separately so important metadata is searchable.
                    fields = parse_place_fields(text)
                    latitude, longitude = parse_coordinates(fields)

                    documents.append(
                        {
                            "text": text,
                            "city": city,
                            "source": text_path.name,
                            # Fall back to the filename if the Name field is missing.
                            "name": fields.get("Name", text_path.stem),
                            "category": fields.get("Category", "Not listed"),
                            "tourist_priority": fields.get("Tourist Priority", "Default"),
                            "latitude": latitude,
                            "longitude": longitude,
                        }
                    )

            except Exception as error:
                print("Could not read:", text_path.name)
                print(error)

    return documents


def create_document_chunks(documents):
    """
    Split each document and create the metadata stored with every chunk.

    The chunk and metadata lists stay in the same order so each chunk keeps the
    correct city, category, source, and coordinates.
    """

    document_chunks = []
    document_metadata = []

    for document in documents:
        chunks = split_text_into_chunks(
            document["text"],
            chunk_size=150,
            overlap=30,
        )

        # From class notes: keep a chunk number with each piece of text.
        for chunk_number, chunk in enumerate(chunks):
            document_chunks.append(chunk)

            # My additions: travel metadata used for city and category filters.
            chunk_metadata = {
                "city": document["city"],
                "source": document["source"],
                "chunk_number": chunk_number,
                "name": document.get("name", document["source"]),
                "category": document.get("category", "Not listed"),
                "tourist_priority": document.get("tourist_priority", "Default"),
            }

            # ChromaDB metadata cannot contain None, so coordinates are optional.
            latitude = document.get("latitude")
            longitude = document.get("longitude")

            if latitude is not None and longitude is not None:
                chunk_metadata["latitude"] = latitude
                chunk_metadata["longitude"] = longitude

            document_metadata.append(chunk_metadata)

    return document_chunks, document_metadata


def build_knowledge_base():
    """Build the ChromaDB collection from the current tourism files.

    This runs the full pipeline: load, chunk, embed, assign IDs, and save. It
    should be run again whenever the source tourism files change.
    """

    documents = load_documents()

    document_chunks, document_metadata = create_document_chunks(documents)

    # Avoid sending an empty batch to the embedding model or database.
    if not document_chunks:
        print("No readable document text was found.")
        return

    print("Creating embeddings...")

    # From class notes: create embeddings for all chunks.
    chunk_embeddings = create_embeddings(document_chunks)

    # From class notes: give every chunk a unique ID before storing it.
    chunk_ids = [
        f"travel_chunk_{index}"
        for index in range(len(document_chunks))
    ]

    # All four lists use the same order, so each ID, chunk, embedding, and
    # metadata dictionary stay connected.
    add_documents(
        documents=document_chunks,
        embeddings=chunk_embeddings,
        metadata=document_metadata,
        ids=chunk_ids,
    )

    print(len(chunk_embeddings), "chunk embeddings created.")
    print("Collection size:", database_count())


def retrieve_context(question, number_of_results=3):
    """
    Return the stored chunks that are most relevant to a question.
    """

    # From class notes: create the question embedding and retrieve the top chunks.
    question_embedding = create_query_embeddings(question)

    results = search_database(
        question_embedding,
        top_k=number_of_results,
    )

    # ChromaDB supports batches of queries. This function sends one, so the
    # results we need are in the first nested list.
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return documents, metadatas, distances

#-----------------------------------------------
# This is the meat and potatoes for geolocation!
#------------------------------------------------
def haversine(latitude_1, longitude_1, latitude_2, longitude_2, unit="km"):
    """
    Calculate the surface distance between two latitude/longitude points.

    The formula treats Earth as a sphere, which is accurate enough for finding
    nearby tourist places without calling an outside mapping service.
    Math is hard!
    """

    if unit not in ("km", "miles"):
        raise ValueError('unit must be either "km" or "miles"')

    # Average Earth radius in both supported units.
    earth_radius_km = 6371.0
    earth_radius_miles = 3958.8

    # Python's trig functions use radians.
    lat1 = math.radians(latitude_1)
    lon1 = math.radians(longitude_1)
    lat2 = math.radians(latitude_2)
    lon2 = math.radians(longitude_2)

    # Difference between the two points in radians.
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    # Main haversine calculation. ``c`` is the central angle between the points.
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Convert the central angle into distance using the requested radius.
    radius = earth_radius_miles if unit == "miles" else earth_radius_km
    return radius * c


def resolve_search_center(anchor_location, metadatas):
    """Find the coordinates and city for the user's anchor location.

    Saved tourism places are checked first. Neighborhoods and districts that
    are not in the database use a single cached OpenStreetMap geocoding lookup.
    """

    search_name = anchor_location.strip().lower()
    comparison_name = re.sub(r"^(?:the|a|an)\s+", "", search_name)
    places_with_coordinates = [
        metadata
        for metadata in metadatas
        if metadata.get("latitude") is not None
        and metadata.get("longitude") is not None
    ]

    # Exact database matches are the most reliable and do not need internet.
    for metadata in places_with_coordinates:
        stored_name = re.sub(
            r"^(?:the|a|an)\s+",
            "",
            metadata.get("name", "").lower(),
        )
        if stored_name == comparison_name:
            return (
                metadata["latitude"],
                metadata["longitude"],
                metadata.get("city"),
            )

    # A close spelling match handles small differences such as punctuation.
    names = {
        re.sub(r"^(?:the|a|an)\s+", "", metadata.get("name", "").lower()): metadata
        for metadata in places_with_coordinates
        if metadata.get("name")
    }
    close_names = difflib.get_close_matches(comparison_name, names, n=1, cutoff=0.9)
    if close_names:
        metadata = names[close_names[0]]
        return metadata["latitude"], metadata["longitude"], metadata.get("city")

    # Check the local cache before making a request to the public geocoder.
    try:
        cache = json.loads(geocode_cache_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}

    if search_name in cache:
        saved = cache[search_name]
        print("Anchor coordinates © OpenStreetMap contributors, found with Nominatim.")
        return saved["latitude"], saved["longitude"], saved["city"]

    global last_geocode_request
    wait_time = 1.0 - (time.monotonic() - last_geocode_request)
    if wait_time > 0:
        time.sleep(wait_time)

    query = urlencode(
        {
            "q": f"{anchor_location}, USA",
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": 1,
        }
    )
    request = Request(
        f"https://nominatim.openstreetmap.org/search?{query}",
        headers={"User-Agent": "NateTourismRAG/1.0 (student travel project)"},
    )

    try:
        with urlopen(request, timeout=10) as response:
            matches = json.load(response)
        last_geocode_request = time.monotonic()
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    if not matches:
        return None

    match = matches[0]
    address = match.get("address", {})
    location_text = " ".join(
        str(value).lower()
        for value in [
            address.get("city"),
            address.get("town"),
            address.get("county"),
            address.get("state"),
            match.get("display_name"),
        ]
        if value
    )
    city = next(
        (
            project_city
            for name, project_city in supported_city_names.items()
            if name in location_text
        ),
        None,
    )
    if city is None:
        return None

    saved = {
        "latitude": float(match["lat"]),
        "longitude": float(match["lon"]),
        "city": city,
    }
    cache[search_name] = saved
    geocode_cache_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    print("Anchor coordinates © OpenStreetMap contributors, found with Nominatim.")
    return saved["latitude"], saved["longitude"], saved["city"]


def find_places_near(
    anchor_location,
    categories=None,
    radius=1.0,
    unit="miles",
    limit=10,
):
    """
    Find filtered places near an anchor location.

    Example calls:

        find_places_near("Fenway Park", categories="bars")
        find_places_near("Copley Square", categories=["attractions", "parks"])
        find_places_near("Downtown Boston", categories=["historic", "attractions"])
        find_places_near("National Mall", categories="museums")

    Pretty cool right!
    """

    results = get_all_documents()
    documents = results["documents"]
    metadatas = results["metadatas"]
    search_name = anchor_location.strip().lower()

    search_center = resolve_search_center(anchor_location, metadatas)
    if search_center is None:
        print(f"Could not find coordinates for '{anchor_location}'.")
        return []
    center_latitude, center_longitude, city = search_center

    # Accept one category, several categories, or no category filter at all.
    if isinstance(categories, str):
        categories = [categories]
    allowed_categories = {
        category_aliases.get(category.lower(), category.lower())
        for category in (categories or [])
    }

    nearby_places = []
    seen_places = set()

    for document_text, metadata in zip(documents, metadatas):
        latitude = metadata.get("latitude")
        longitude = metadata.get("longitude")
        name = metadata.get("name", metadata.get("source", "Unknown"))
        category = metadata.get("category", "Not listed").lower()
        place_key = (metadata.get("city"), name.lower())

        if latitude is None or longitude is None:
            continue
        if metadata.get("city", "").lower() != city.lower():
            continue
        if allowed_categories and category not in allowed_categories:
            continue
        if name.lower() == search_name or place_key in seen_places:
            continue

        distance = haversine(
            center_latitude,
            center_longitude,
            latitude,
            longitude,
            unit=unit,
        )

        if distance <= radius:
            seen_places.add(place_key)
            nearby_places.append(
                {
                    "name": name,
                    "city": metadata.get("city"),
                    "category": metadata.get("category"),
                    "tourist_priority": metadata.get("tourist_priority", "Default"),
                    "distance": round(distance, 3),
                    "unit": unit,
                    "text": document_text,
                }
            )

    # Distance matters most here; Tourist Priority is still available to the LLM.
    nearby_places.sort(key=lambda place: place["distance"])
    return nearby_places[:limit]


def find_places_from_question(question, radius=1.0, unit="miles", limit=10):
    """Turn a basic nearby-place question into a location search.

    The wording before "near," "around," "by," "at," or "in" describes what
    the user wants. The wording after it becomes the anchor location.
    """

    question_text = question.strip().lower()
    location_match = re.search(
        r"\b(?:near|around|by|at|in)\b\s+(.+?)[?.!]*$",
        question_text,
    )
    if location_match is None:
        print("I could not tell which anchor location the question meant.")
        return []

    anchor_location = location_match.group(1).strip()
    request_text = question_text[:location_match.start()]
    categories = next(
        (
            values
            for phrase, values in question_categories.items()
            if re.search(rf"\b{re.escape(phrase)}\b", request_text)
        ),
        None,
    )

    return find_places_near(
        anchor_location=anchor_location,
        categories=categories,
        radius=radius,
        unit=unit,
        limit=limit,
    )
