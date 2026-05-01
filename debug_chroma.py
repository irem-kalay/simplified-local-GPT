import chromadb
from sentence_transformers import SentenceTransformer
import warnings

# PyTorch uyarılarını gizle
warnings.filterwarnings("ignore")

print("Modeller yükleniyor...")
# Veritabanı ve embedding modelini bağla
chroma_client = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma_client.get_collection(name="wikipedia_entities")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Test edilecek soru
query = "Compare the Eiffel Tower and the Statue of Liberty"
print(f"\nSorgu: '{query}'")

# Soruyu vektöre (sayılara) çevir
query_embedding = model.encode([query])[0].tolist()

# Havuzdan ilk 100 sonucu getir
n_results_to_fetch = 500
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=n_results_to_fetch
)

# Sonuçların hangi konulardan (entity) geldiğini sayalım
print(f"\n--- İlk {n_results_to_fetch} Sonucun Dağılımı ---")
for i, meta in enumerate(results["metadatas"][0]):
    entity_name = meta.get('entity_name', 'Unknown')
    chunk_index = meta.get('chunk_index', 'Unknown')
    print(f"Sıra {i+1}: {entity_name} (Chunk {chunk_index})")