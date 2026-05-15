
# ingest.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed.sparse import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    NamedSparseVector,
    NamedVector,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

load_dotenv()

# ─── STEP 1: LOAD DOCUMENTS ───────────────────────────────────────────
# Block 6: PyPDFLoader → returns List[Document]
# Each Document has page_content (text) + metadata (source, page number)

print("📂 Loading documents...")

loader = DirectoryLoader(
    "data/laws/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()
print(f"   Loaded {len(documents)} pages from {len(set(d.metadata['source'] for d in documents))} files")

# ─── STEP 2: TAG METADATA ─────────────────────────────────────────────
# Block 6: metadata filtering — tag every chunk with its source act
# Without this, retriever searches ALL acts and finds wrong content
# "What is Section 302?" might return Consumer Protection Act chunks

ACT_NAMES = {
    "ipc": "IPC",
    "constitution": "Constitution",
    "rti": "RTI",
    "consumer": "ConsumerProtection",
    "rent": "TNRentControl",
}

for doc in documents:
    source = doc.metadata.get("source", "").lower()
    act = "Unknown"
    for key, name in ACT_NAMES.items():
        if key in source:
            act = name
            break
    doc.metadata["act"] = act  # ← this is what filter={"act": "IPC"} reads later

print("   Tagged all documents with act metadata")

# ─── STEP 3: CHUNK DOCUMENTS ──────────────────────────────────────────
# Block 6: structure-aware splitting for legal documents
# Why these separators? Legal acts have Sections, Articles, Clauses
# We split on those boundaries first — each chunk = one complete legal section
# chunk_overlap=100 prevents mid-sentence cuts (the answer split problem)

print("✂️  Chunking documents...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " "]
)

chunks = splitter.split_documents(documents)
chunks = [
    c for c in chunks
    if len(c.page_content.splitlines()) > 8
    or len(c.page_content) > 300
]
print(f"   Created {len(chunks)} chunks from {len(documents)} pages")
print(f"   Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

# ─── STEP 4: CREATE EMBEDDINGS ────────────────────────────────────────
# Block 6: embedding model converts text → vector of floats
# MiniLM = free, runs locally on your CPU, good quality
# 384-dimensional vectors — each chunk becomes [0.21, -0.54, 0.88, ...]
# Critical rule: same model for indexing AND querying
# Different model = different vector space = garbage retrieval

print("🧠 Loading embedding model...")
print("   (downloads ~90MB first time, cached after)")

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},   # your CPU handles this
    encode_kwargs={"normalize_embeddings": True}
)

print("   Embedding model loaded")

print("🧠 Loading sparse embedding model...")
print("   (downloads SPLADE model first time, cached after)")

sparse_embedder = SparseTextEmbedding(model_name="prithivida/Splade_PP_ENv1")

print("   Sparse embedding model loaded")

# ─── STEP 5: STORE IN QDRANT ──────────────────────────────────────────
# Block 6: vector store — stores vectors + original text together
# Qdrant Cloud — stores vectors + original text together
# from_documents() does: embed each chunk → store vector + text + metadata
# This is what gets searched at query time

print("📦 Storing vectors in Qdrant...")

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

# Create collection if it doesn't exist
collection_name = os.getenv("QDRANT_COLLECTION", "vakil_legal")

existing = [c.name for c in client.get_collections().collections]
if collection_name in existing:
    print(f"   Collection '{collection_name}' exists — deleting and recreating")
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "dense": VectorParams(
            size=384,           # MiniLM output dimension
            distance=Distance.COSINE  # cosine similarity for semantic search
        )
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams()
    }
)

# Embed all chunks and store — this takes a few minutes
batch_size = 100
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    texts = [doc.page_content for doc in batch]
    dense_vectors = embedder.embed_documents(texts)
    sparse_vectors = list(sparse_embedder.embed(texts))

    points = []
    for j, doc in enumerate(batch):
        dense_vector = NamedVector(name="dense", vector=dense_vectors[j])
        sparse_vector = NamedSparseVector(
            name="sparse",
            vector=SparseVector(
                indices=[int(index) for index in sparse_vectors[j].indices],
                values=[float(value) for value in sparse_vectors[j].values],
            )
        )
        points.append(
            PointStruct(
                id=i + j,
                vector={
                    dense_vector.name: dense_vector.vector,
                    sparse_vector.name: sparse_vector.vector,
                },
                payload={
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                },
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points,
    )
    print(f"   Uploaded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

print(f"✅ Indexed {len(chunks)} chunks into Qdrant collection '{collection_name}'")
print(f"   Qdrant URL: {os.getenv('QDRANT_URL')}")

# ─── STEP 6: VERIFY ───────────────────────────────────────────────────
# Quick sanity check — does retrieval actually work?

print("\n🔍 Testing retrieval...")

# retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
# test_results = retriever.invoke("What is punishment for murder?")

# for i, doc in enumerate(test_results):
#     print(f"\n   Result {i+1}:")
#     print(f"   Act: {doc.metadata.get('act', 'unknown')}")
#     print(f"   Source: {doc.metadata.get('source', 'unknown')}")
#     print(f"   Preview: {doc.page_content[:150]}...")

# print("\n✅ Ingest complete. Run main.py to start the server.")
