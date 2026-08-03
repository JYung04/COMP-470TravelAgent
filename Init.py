#initializes embedding only needs to run when data is added or on initial startup

from rag import build_knowledge_base

def setup():
    build_knowledge_base()