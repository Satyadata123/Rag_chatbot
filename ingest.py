import shutil
import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from common import PDF_PATH, DB_PATH, COLLECTION, get_embeddings

MAX_CHARS = 1800   # about 450 tokens
OVERLAP = 300      # about 75 tokens

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=False,       # keep the heading text inside the chunk
)
size_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHARS, chunk_overlap=OVERLAP)


def project_of(heading: str) -> str:
    h = heading.lower()
    if "fraud" in h:
        return "fraud"
    if "churn" in h:
        return "churn"
    return "general"


def heading_of(doc: Document) -> str:
    for key in ("h3", "h2", "h1"):          # most specific heading wins
        if key in doc.metadata:
            return doc.metadata[key].strip("* ").strip()
    return "Introduction"


def load_and_split():
    pages = pymupdf4llm.to_markdown(PDF_PATH, page_chunks=True)   # 1. LOAD
    final_docs = []

    for page_no, page in enumerate(pages, start=1):
        sections = header_splitter.split_text(page["text"])        # 2. SPLIT by heading
        for sec in sections:
            heading = heading_of(sec)
            if "sample questions" in heading.lower():
                continue    # test set, not a knowledge source
            keep_whole = "metric" in heading.lower() or len(sec.page_content) <= MAX_CHARS
            parts = [sec] if keep_whole else size_splitter.split_documents([sec])   # 3. then by size

            for part in parts:
                if not part.page_content.lstrip().startswith("#"):
                    part.page_content = f"# {heading}\n{part.page_content}"
                part.metadata = {
                    "page": page_no,
                    "heading": heading,
                    "project": project_of(heading),
                }
                final_docs.append(part)
    return final_docs


def main():
    docs = load_and_split()
    print(f"Created {len(docs)} chunks")

    shutil.rmtree(DB_PATH, ignore_errors=True)       # rebuild from scratch
    Chroma.from_documents(                            # 4. EMBED + STORE
        documents=docs,
        embedding=get_embeddings(),
        collection_name=COLLECTION,
        persist_directory=DB_PATH,
    )
    print("Done. Vector DB saved in ./chroma_db")

    for d in docs[:5]:
        print("\n---", d.metadata)
        print(d.page_content[:200])


if __name__ == "__main__":
    main()