import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QdrantManager")

class QdrantManager:
    """Manager for Qdrant Vector Database"""
    
    def __init__(self, host="localhost", port=6333, collection_name="knowledge"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = QdrantClient(host=self.host, port=self.port)
        logger.info(f"Connected to Qdrant at {self.host}:{self.port}")

    def create_collection(self, vector_size=512, distance=Distance.COSINE):
        """Creates or recreates the collection"""
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            logger.info(f"Collection '{self.collection_name}' created (dim={vector_size})")
            return True
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False

    def upsert_documents(self, chunks, embeddings, source_name="unknown"):
        """Upserts document chunks and their embeddings into Qdrant"""
        try:
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Qdrant IDs can be integers or UUID strings
                # We'll generate a unique ID based on source and index if needed, 
                # but for simplicity let's use a hash or a counter
                point_id = hash(f"{source_name}_{i}") & 0xFFFFFFFFFFFFFFFF
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    payload={
                        "text": chunk,
                        "source": source_name,
                        "chunk_index": i
                    }
                ))
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Upserted {len(points)} points from '{source_name}' to '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error upserting documents: {e}")
            return False

    def search(self, query_vector, limit=3):
        """Searches for the most similar chunks in Qdrant"""
        try:
            # query_points is the new API for search in qdrant-client 1.11+
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector,
                limit=limit
            )
            
            results = []
            for hit in search_result.points:
                results.append({
                    "text": hit.payload["text"],
                    "source": hit.payload["source"],
                    "score": hit.score
                })
            
            return results
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            return []

if __name__ == "__main__":
    # Test the manager
    manager = QdrantManager()
    
    # Example vector (512 dimensions for distiluse-base-multilingual-cased-v2)
    test_vector = np.random.rand(512).astype(np.float32)
    
    # 1. Create collection
    manager.create_collection(vector_size=512)
    
    # 2. Upsert dummy data
    manager.upsert_documents(
        chunks=["Vika is an AI assistant.", "It routes requests between LLM providers."],
        embeddings=[test_vector, test_vector],
        source_name="test_doc"
    )
    
    # 3. Search
    results = manager.search(test_vector, limit=1)
    if results:
        print(f"Search successful! Top match: {results[0]['text']} (Score: {results[0]['score']:.4f})")
    else:
        print("Search failed or returned no results.")
