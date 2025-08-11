#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Vector Memory System
ChromaDB/Weaviate-based vector storage for agent experiences and knowledge
"""

import asyncio
import logging
import uuid
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import numpy as np

# Vector database imports (with fallbacks)
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    weaviate = None

# Embedding imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

from agents.base_agent import Experience, Team, Role

logger = logging.getLogger(__name__)

class VectorDBType(Enum):
    """Supported vector database types"""
    CHROMADB = "chromadb"
    WEAVIATE = "weaviate"
    MEMORY = "memory"  # In-memory fallback

@dataclass
class MemoryCluster:
    """Clustered memories with semantic similarity"""
    cluster_id: str
    cluster_name: str
    center_embedding: List[float]
    experiences: List[str]  # Experience IDs
    similarity_threshold: float
    created_at: datetime
    last_updated: datetime
    metadata: Dict[str, Any]

@dataclass
class MemoryQuery:
    """Query structure for memory retrieval"""
    query_text: str
    agent_id: Optional[str] = None
    team: Optional[Team] = None
    role: Optional[Role] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    similarity_threshold: float = 0.7
    max_results: int = 10
    include_metadata: bool = True

@dataclass
class MemoryResult:
    """Result from memory query"""
    experience_id: str
    experience: Experience
    similarity_score: float
    embedding: List[float]
    metadata: Dict[str, Any]

class VectorMemorySystem:
    """
    Vector-based memory system for storing and retrieving agent experiences.
    
    Features:
    - Semantic search using embeddings
    - Memory clustering for organization
    - Role-specific memory isolation
    - Temporal memory management
    - Experience similarity matching
    """
    
    def __init__(self, 
                 db_type: VectorDBType = VectorDBType.CHROMADB,
                 db_path: str = "./memory_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 collection_name: str = "agent_experiences"):
        
        self.db_type = db_type
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        
        # Database clients
        self.chroma_client = None
        self.weaviate_client = None
        self.collection = None
        
        # Embedding model
        self.embedding_model = None
        
        # In-memory fallback
        self.memory_store: Dict[str, Dict[str, Any]] = {}
        self.embeddings_store: Dict[str, List[float]] = {}
        
        # Memory clusters
        self.clusters: Dict[str, MemoryCluster] = {}
        
        # Configuration
        self.max_memory_size = 10000
        self.clustering_threshold = 0.8
        self.cleanup_interval = timedelta(hours=24)
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the vector memory system"""
        try:
            self.logger.info(f"Initializing vector memory system with {self.db_type.value}")
            
            # Initialize embedding model
            await self._initialize_embedding_model()
            
            # Initialize vector database
            if self.db_type == VectorDBType.CHROMADB and CHROMADB_AVAILABLE:
                await self._initialize_chromadb()
            elif self.db_type == VectorDBType.WEAVIATE and WEAVIATE_AVAILABLE:
                await self._initialize_weaviate()
            else:
                self.logger.warning("Vector database not available, using in-memory storage")
                self.db_type = VectorDBType.MEMORY
            
            # Load existing clusters
            await self._load_clusters()
            
            self.initialized = True
            self.logger.info("Vector memory system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector memory system: {e}")
            # Fallback to in-memory storage
            self.db_type = VectorDBType.MEMORY
            self.initialized = True
    
    async def _initialize_embedding_model(self) -> None:
        """Initialize the sentence transformer model"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.embedding_model = await loop.run_in_executor(
                    None, 
                    SentenceTransformer, 
                    self.embedding_model_name
                )
                self.logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            else:
                self.logger.warning("SentenceTransformers not available, using simple embeddings")
                
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    async def _initialize_chromadb(self) -> None:
        """Initialize ChromaDB client and collection"""
        try:
            # Create ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.collection_name
                )
                self.logger.info(f"Loaded existing ChromaDB collection: {self.collection_name}")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Agent experiences and memories"}
                )
                self.logger.info(f"Created new ChromaDB collection: {self.collection_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    async def _initialize_weaviate(self) -> None:
        """Initialize Weaviate client"""
        try:
            # This would be implemented with actual Weaviate configuration
            self.logger.info("Weaviate initialization would be implemented here")
            raise NotImplementedError("Weaviate support not yet implemented")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Weaviate: {e}")
            raise
    
    async def store_experience(self, agent_id: str, experience: Experience) -> str:
        """
        Store an agent experience in vector memory
        
        Args:
            agent_id: ID of the agent
            experience: Experience to store
            
        Returns:
            str: Stored experience ID
        """
        try:
            # Create embedding for the experience
            experience_text = self._experience_to_text(experience)
            embedding = await self._create_embedding(experience_text)
            
            # Create metadata
            metadata = {
                "agent_id": agent_id,
                "timestamp": experience.timestamp.isoformat(),
                "success": experience.success,
                "confidence": experience.confidence_score,
                "team": experience.action_taken.target if hasattr(experience.action_taken, 'target') else "unknown",
                "action_type": experience.action_taken.action_type if hasattr(experience.action_taken, 'action_type') else "unknown"
            }
            
            # Store in vector database
            if self.db_type == VectorDBType.CHROMADB and self.collection:
                await self._store_in_chromadb(experience.experience_id, experience_text, embedding, metadata)
            else:
                # Store in memory fallback
                self.memory_store[experience.experience_id] = {
                    "experience": asdict(experience),
                    "text": experience_text,
                    "metadata": metadata
                }
                self.embeddings_store[experience.experience_id] = embedding
            
            # Update clusters
            await self._update_clusters(experience.experience_id, embedding, metadata)
            
            self.logger.debug(f"Stored experience {experience.experience_id} for agent {agent_id}")
            return experience.experience_id
            
        except Exception as e:
            self.logger.error(f"Failed to store experience: {e}")
            raise
    
    async def _store_in_chromadb(self, experience_id: str, text: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Store experience in ChromaDB"""
        try:
            self.collection.add(
                ids=[experience_id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata]
            )
        except Exception as e:
            self.logger.error(f"Failed to store in ChromaDB: {e}")
            raise
    
    async def retrieve_similar_experiences(self, query: MemoryQuery) -> List[MemoryResult]:
        """
        Retrieve experiences similar to the query
        
        Args:
            query: Memory query parameters
            
        Returns:
            List[MemoryResult]: Similar experiences with scores
        """
        try:
            # Create embedding for query
            query_embedding = await self._create_embedding(query.query_text)
            
            # Search in vector database
            if self.db_type == VectorDBType.CHROMADB and self.collection:
                results = await self._search_chromadb(query, query_embedding)
            else:
                # Search in memory fallback
                results = await self._search_memory(query, query_embedding)
            
            # Filter and sort results
            filtered_results = self._filter_results(results, query)
            
            self.logger.debug(f"Retrieved {len(filtered_results)} similar experiences")
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve similar experiences: {e}")
            return []
    
    async def _search_chromadb(self, query: MemoryQuery, query_embedding: List[float]) -> List[MemoryResult]:
        """Search ChromaDB for similar experiences"""
        try:
            # Build where clause for filtering
            where_clause = {}
            if query.agent_id:
                where_clause["agent_id"] = query.agent_id
            if query.team:
                where_clause["team"] = query.team.value
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=query.max_results,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances", "embeddings"] if query.include_metadata else ["documents", "distances"]
            )
            
            # Convert to MemoryResult objects
            memory_results = []
            for i in range(len(results["ids"][0])):
                experience_id = results["ids"][0][i]
                similarity_score = 1.0 - results["distances"][0][i]  # Convert distance to similarity
                
                if similarity_score >= query.similarity_threshold:
                    # Reconstruct experience (simplified for demo)
                    experience_data = {
                        "experience_id": experience_id,
                        "agent_id": results["metadatas"][0][i]["agent_id"],
                        "timestamp": datetime.fromisoformat(results["metadatas"][0][i]["timestamp"]),
                        "success": results["metadatas"][0][i]["success"],
                        "confidence_score": results["metadatas"][0][i]["confidence"]
                    }
                    
                    memory_result = MemoryResult(
                        experience_id=experience_id,
                        experience=experience_data,  # Simplified
                        similarity_score=similarity_score,
                        embedding=results["embeddings"][0][i] if "embeddings" in results else [],
                        metadata=results["metadatas"][0][i]
                    )
                    memory_results.append(memory_result)
            
            return memory_results
            
        except Exception as e:
            self.logger.error(f"Failed to search ChromaDB: {e}")
            return []
    
    async def _search_memory(self, query: MemoryQuery, query_embedding: List[float]) -> List[MemoryResult]:
        """Search in-memory storage for similar experiences"""
        try:
            results = []
            
            for exp_id, exp_data in self.memory_store.items():
                # Calculate similarity
                exp_embedding = self.embeddings_store.get(exp_id, [])
                if not exp_embedding:
                    continue
                
                similarity = self._calculate_similarity(query_embedding, exp_embedding)
                
                if similarity >= query.similarity_threshold:
                    # Apply filters
                    metadata = exp_data["metadata"]
                    if query.agent_id and metadata.get("agent_id") != query.agent_id:
                        continue
                    if query.team and metadata.get("team") != query.team.value:
                        continue
                    
                    memory_result = MemoryResult(
                        experience_id=exp_id,
                        experience=exp_data["experience"],
                        similarity_score=similarity,
                        embedding=exp_embedding,
                        metadata=metadata
                    )
                    results.append(memory_result)
            
            # Sort by similarity
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results[:query.max_results]
            
        except Exception as e:
            self.logger.error(f"Failed to search memory: {e}")
            return []
    
    async def cluster_memories(self, agent_id: str, time_window: timedelta) -> List[MemoryCluster]:
        """
        Cluster agent memories by semantic similarity
        
        Args:
            agent_id: Agent to cluster memories for
            time_window: Time window for clustering
            
        Returns:
            List[MemoryCluster]: Memory clusters
        """
        try:
            # Get recent experiences for agent
            end_time = datetime.now()
            start_time = end_time - time_window
            
            query = MemoryQuery(
                query_text="",  # Empty query to get all
                agent_id=agent_id,
                time_range=(start_time, end_time),
                similarity_threshold=0.0,  # Get all experiences
                max_results=1000
            )
            
            experiences = await self.retrieve_similar_experiences(query)
            
            if len(experiences) < 2:
                return []
            
            # Simple clustering algorithm (could be improved with proper clustering)
            clusters = []
            used_experiences = set()
            
            for i, exp1 in enumerate(experiences):
                if exp1.experience_id in used_experiences:
                    continue
                
                # Create new cluster
                cluster_experiences = [exp1.experience_id]
                cluster_embeddings = [exp1.embedding]
                used_experiences.add(exp1.experience_id)
                
                # Find similar experiences
                for j, exp2 in enumerate(experiences[i+1:], i+1):
                    if exp2.experience_id in used_experiences:
                        continue
                    
                    similarity = self._calculate_similarity(exp1.embedding, exp2.embedding)
                    if similarity >= self.clustering_threshold:
                        cluster_experiences.append(exp2.experience_id)
                        cluster_embeddings.append(exp2.embedding)
                        used_experiences.add(exp2.experience_id)
                
                # Create cluster if it has multiple experiences
                if len(cluster_experiences) > 1:
                    center_embedding = np.mean(cluster_embeddings, axis=0).tolist()
                    
                    cluster = MemoryCluster(
                        cluster_id=str(uuid.uuid4()),
                        cluster_name=f"cluster_{len(clusters)+1}",
                        center_embedding=center_embedding,
                        experiences=cluster_experiences,
                        similarity_threshold=self.clustering_threshold,
                        created_at=datetime.now(),
                        last_updated=datetime.now(),
                        metadata={"agent_id": agent_id, "size": len(cluster_experiences)}
                    )
                    clusters.append(cluster)
            
            self.logger.info(f"Created {len(clusters)} memory clusters for agent {agent_id}")
            return clusters
            
        except Exception as e:
            self.logger.error(f"Failed to cluster memories: {e}")
            return []
    
    async def update_tactical_knowledge(self, agent_id: str, tactic: Dict[str, Any]) -> None:
        """Update tactical knowledge for an agent"""
        try:
            # Create a synthetic experience for the tactic
            tactic_text = json.dumps(tactic, indent=2)
            embedding = await self._create_embedding(tactic_text)
            
            metadata = {
                "agent_id": agent_id,
                "type": "tactical_knowledge",
                "timestamp": datetime.now().isoformat(),
                "tactic_type": tactic.get("type", "unknown")
            }
            
            tactic_id = f"tactic_{uuid.uuid4()}"
            
            # Store tactical knowledge
            if self.db_type == VectorDBType.CHROMADB and self.collection:
                await self._store_in_chromadb(tactic_id, tactic_text, embedding, metadata)
            else:
                self.memory_store[tactic_id] = {
                    "tactic": tactic,
                    "text": tactic_text,
                    "metadata": metadata
                }
                self.embeddings_store[tactic_id] = embedding
            
            self.logger.debug(f"Updated tactical knowledge for agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to update tactical knowledge: {e}")
    
    async def get_role_specific_memories(self, agent_id: str, role: Role) -> List[MemoryResult]:
        """Get memories specific to an agent role"""
        try:
            query = MemoryQuery(
                query_text=f"role:{role.value}",
                agent_id=agent_id,
                role=role,
                similarity_threshold=0.5,
                max_results=50
            )
            
            return await self.retrieve_similar_experiences(query)
            
        except Exception as e:
            self.logger.error(f"Failed to get role-specific memories: {e}")
            return []
    
    async def _create_embedding(self, text: str) -> List[float]:
        """Create embedding for text"""
        try:
            if self.embedding_model and SENTENCE_TRANSFORMERS_AVAILABLE:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    None,
                    self.embedding_model.encode,
                    text
                )
                return embedding.tolist()
            else:
                # Simple fallback embedding (hash-based)
                return self._simple_embedding(text)
                
        except Exception as e:
            self.logger.error(f"Failed to create embedding: {e}")
            return self._simple_embedding(text)
    
    def _simple_embedding(self, text: str, dim: int = 384) -> List[float]:
        """Create simple hash-based embedding as fallback"""
        # Simple hash-based embedding for fallback
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to float vector
        embedding = []
        for i in range(0, len(hash_bytes), 4):
            chunk = hash_bytes[i:i+4]
            if len(chunk) == 4:
                val = int.from_bytes(chunk, 'big') / (2**32)
                embedding.append(val)
        
        # Pad or truncate to desired dimension
        while len(embedding) < dim:
            embedding.extend(embedding[:min(len(embedding), dim - len(embedding))])
        
        return embedding[:dim]
    
    def _experience_to_text(self, experience: Experience) -> str:
        """Convert experience to searchable text"""
        try:
            text_parts = [
                f"Agent: {experience.agent_id}",
                f"Success: {experience.success}",
                f"Confidence: {experience.confidence_score}",
            ]
            
            # Add action information if available
            if hasattr(experience, 'action_taken') and experience.action_taken:
                text_parts.append(f"Action: {getattr(experience.action_taken, 'primary_action', 'unknown')}")
                text_parts.append(f"Type: {getattr(experience.action_taken, 'action_type', 'unknown')}")
            
            # Add outcome information
            if hasattr(experience, 'outcome') and experience.outcome:
                text_parts.append(f"Outcome: {getattr(experience.outcome, 'outcome', 'unknown')}")
            
            # Add lessons learned
            if experience.lessons_learned:
                text_parts.append(f"Lessons: {' '.join(experience.lessons_learned)}")
            
            return " | ".join(text_parts)
            
        except Exception as e:
            self.logger.error(f"Failed to convert experience to text: {e}")
            return f"Experience {experience.experience_id}"
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            if not embedding1 or not embedding2:
                return 0.0
            
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def _filter_results(self, results: List[MemoryResult], query: MemoryQuery) -> List[MemoryResult]:
        """Filter and sort memory results"""
        try:
            filtered = []
            
            for result in results:
                # Apply time range filter
                if query.time_range:
                    exp_time = result.metadata.get("timestamp")
                    if exp_time:
                        exp_datetime = datetime.fromisoformat(exp_time)
                        if not (query.time_range[0] <= exp_datetime <= query.time_range[1]):
                            continue
                
                # Apply similarity threshold
                if result.similarity_score < query.similarity_threshold:
                    continue
                
                filtered.append(result)
            
            # Sort by similarity score
            filtered.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return filtered[:query.max_results]
            
        except Exception as e:
            self.logger.error(f"Failed to filter results: {e}")
            return results
    
    async def _update_clusters(self, experience_id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Update memory clusters with new experience"""
        try:
            # Find best matching cluster
            best_cluster = None
            best_similarity = 0.0
            
            for cluster in self.clusters.values():
                similarity = self._calculate_similarity(embedding, cluster.center_embedding)
                if similarity > best_similarity and similarity >= cluster.similarity_threshold:
                    best_similarity = similarity
                    best_cluster = cluster
            
            if best_cluster:
                # Add to existing cluster
                best_cluster.experiences.append(experience_id)
                best_cluster.last_updated = datetime.now()
                
                # Update cluster center (simple average)
                all_embeddings = [best_cluster.center_embedding]
                # In a real implementation, we'd load all experience embeddings
                # For now, just update with new embedding
                new_center = np.mean([best_cluster.center_embedding, embedding], axis=0)
                best_cluster.center_embedding = new_center.tolist()
            
        except Exception as e:
            self.logger.error(f"Failed to update clusters: {e}")
    
    async def _load_clusters(self) -> None:
        """Load existing memory clusters"""
        try:
            # In a real implementation, this would load from persistent storage
            self.logger.debug("Loading memory clusters (placeholder)")
            
        except Exception as e:
            self.logger.error(f"Failed to load clusters: {e}")
    
    async def cleanup_old_memories(self, max_age: timedelta) -> int:
        """Clean up old memories beyond max age"""
        try:
            cutoff_time = datetime.now() - max_age
            cleaned_count = 0
            
            if self.db_type == VectorDBType.CHROMADB and self.collection:
                # ChromaDB cleanup would be implemented here
                pass
            else:
                # Clean up in-memory storage
                to_remove = []
                for exp_id, exp_data in self.memory_store.items():
                    timestamp_str = exp_data["metadata"].get("timestamp")
                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        if timestamp < cutoff_time:
                            to_remove.append(exp_id)
                
                for exp_id in to_remove:
                    del self.memory_store[exp_id]
                    if exp_id in self.embeddings_store:
                        del self.embeddings_store[exp_id]
                    cleaned_count += 1
            
            self.logger.info(f"Cleaned up {cleaned_count} old memories")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old memories: {e}")
            return 0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        try:
            if self.db_type == VectorDBType.CHROMADB and self.collection:
                # Get ChromaDB stats
                count = self.collection.count()
                storage_type = "ChromaDB"
            else:
                # Get in-memory stats
                count = len(self.memory_store)
                storage_type = "In-Memory"
            
            return {
                "storage_type": storage_type,
                "total_experiences": count,
                "total_clusters": len(self.clusters),
                "embedding_model": self.embedding_model_name,
                "initialized": self.initialized,
                "db_type": self.db_type.value
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}
    
    async def shutdown(self) -> None:
        """Shutdown the vector memory system"""
        try:
            self.logger.info("Shutting down vector memory system")
            
            # Close database connections
            if self.chroma_client:
                # ChromaDB doesn't need explicit closing
                pass
            
            if self.weaviate_client:
                # Weaviate client closing would be implemented here
                pass
            
            # Clear in-memory storage
            self.memory_store.clear()
            self.embeddings_store.clear()
            self.clusters.clear()
            
            self.initialized = False
            self.logger.info("Vector memory system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during memory system shutdown: {e}")

# Factory function
def create_vector_memory(db_type: VectorDBType = VectorDBType.CHROMADB,
                        db_path: str = "./memory_db",
                        embedding_model: str = "all-MiniLM-L6-v2") -> VectorMemorySystem:
    """Create a vector memory system instance"""
    return VectorMemorySystem(
        db_type=db_type,
        db_path=db_path,
        embedding_model=embedding_model
    )