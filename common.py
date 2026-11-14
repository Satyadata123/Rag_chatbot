from langchain_community.embeddings import FastEmbedEmbeddings

PDF_PATH = "data/Satyam_Singh_RAG_Knowledge_Base.pdf"
DB_PATH = "chroma_db"
COLLECTION = "satyam_kb"


def get_embeddings():
    return FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        cache_dir="models",   # keeps the model in ./models so Docker can cache it
    )