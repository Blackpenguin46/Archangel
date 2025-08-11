#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Memory Manager
Unified interface for vector memory and knowledge base systems
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from .vector_memory import VectorMemorySystem, VectorDBType, MemoryQuery, MemoryResult, MemoryCluster
from .knowledge_base import KnowledgeBase, AttackPattern, DefenseStrategy, TacticType, DefenseTactic
from agents.base_agent import Experience, Team, Role

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Configuration for memory manager"""
    vector_db_type: VectorDBType = VectorDBType.CHROMADB
    vector_db_path: str = "./memory_db"
    knowledge_base_path: str = "./knowledge_base"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_memory_age: timedelta = timedelta(days=30)
    cleanup_interval: timedelta = timedelta(hours=6)
    clustering_enabled: bool = True
    clustering_interval: timedelta = timedelta(hours=12)

class MemoryManager:
    """
    Unified memory management system combining vector memory and knowledge base.
    
    Features:
    - Unified interface for all memory operations
    - Automatic memory clustering and organization
    - Knowledge base integration with MITRE ATT&CK
    - Memory cleanup and maintenance
    - Cross-system search and retrieval
    """
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        
        # Memory systems
        self.vector_memory = VectorMemorySystem(
            db_type=self.config.vector_db_type,
            db_path=self.config.vector_db_path,
            embedding_model=self.config.embedding_model
        )
        
        self.knowledge_base = KnowledgeBase(
            data_dir=self.config.knowledge_base_path
        )
        
        # Background tasks
        self.cleanup_task = None
        self.clustering_task = None
        
        # Statistics
        self.stats = {
            'experiences_stored': 0,
            'queries_processed': 0,
            'clusters_created': 0,
            'knowledge_updates': 0,
            'cleanup_runs': 0
        }
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        self.running = False
    
    async def initialize(self) -> None:
        """Initialize the memory manager"""
        try:
            self.logger.info("Initializing memory manager")
            
            # Initialize subsystems
            await self.vector_memory.initialize()
            await self.knowledge_base.initialize()
            
            # Start background tasks
            if self.config.cleanup_interval:
                self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            if self.config.clustering_enabled and self.config.clustering_interval:
                self.clustering_task = asyncio.create_task(self._clustering_loop())
            
            self.initialized = True
            self.running = True
            
            self.logger.info("Memory manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory manager: {e}")
            raise
    
    async def store_agent_experience(self, agent_id: str, experience: Experience) -> str:
        """
        Store an agent experience in vector memory and update knowledge base
        
        Args:
            agent_id: ID of the agent
            experience: Experience to store
            
        Returns:
            str: Stored experience ID
        """
        try:
            # Store in vector memory
            experience_id = await self.vector_memory.store_experience(agent_id, experience)
            
            # Extract knowledge for knowledge base
            await self._extract_knowledge_from_experience(experience)
            
            # Update statistics
            self.stats['experiences_stored'] += 1
            
            self.logger.debug(f"Stored experience {experience_id} for agent {agent_id}")
            return experience_id
            
        except Exception as e:
            self.logger.error(f"Failed to store agent experience: {e}")
            raise
    
    async def retrieve_similar_experiences(self, 
                                         query_text: str,
                                         agent_id: Optional[str] = None,
                                         team: Optional[Team] = None,
                                         role: Optional[Role] = None,
                                         max_results: int = 10,
                                         similarity_threshold: float = 0.7) -> List[MemoryResult]:
        """
        Retrieve experiences similar to the query
        
        Args:
            query_text: Text query for similarity search
            agent_id: Optional agent ID filter
            team: Optional team filter
            role: Optional role filter
            max_results: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List[MemoryResult]: Similar experiences
        """
        try:
            query = MemoryQuery(
                query_text=query_text,
                agent_id=agent_id,
                team=team,
                role=role,
                similarity_threshold=similarity_threshold,
                max_results=max_results
            )
            
            results = await self.vector_memory.retrieve_similar_experiences(query)
            
            # Update statistics
            self.stats['queries_processed'] += 1
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve similar experiences: {e}")
            return []
    
    async def get_tactical_knowledge(self, 
                                   query: str,
                                   tactics: Optional[List[TacticType]] = None,
                                   knowledge_type: str = "both") -> Dict[str, Any]:
        """
        Get tactical knowledge from knowledge base
        
        Args:
            query: Search query
            tactics: Optional tactic filters
            knowledge_type: "attack", "defense", or "both"
            
        Returns:
            Dict containing attack patterns and/or defense strategies
        """
        try:
            results = {}
            
            if knowledge_type in ["attack", "both"]:
                attack_patterns = await self.knowledge_base.search_attack_patterns(query, tactics)
                results["attack_patterns"] = attack_patterns
            
            if knowledge_type in ["defense", "both"]:
                # Convert TacticType to DefenseTactic for defense search
                defense_tactics = None
                if tactics:
                    # Simple mapping - in practice this would be more sophisticated
                    defense_tactics = [DefenseTactic.DETECT, DefenseTactic.DENY]
                
                defense_strategies = await self.knowledge_base.search_defense_strategies(query, defense_tactics)
                results["defense_strategies"] = defense_strategies
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get tactical knowledge: {e}")
            return {}
    
    async def cluster_agent_memories(self, agent_id: str, time_window: timedelta = None) -> List[MemoryCluster]:
        """
        Cluster memories for a specific agent
        
        Args:
            agent_id: Agent to cluster memories for
            time_window: Time window for clustering (default: 7 days)
            
        Returns:
            List[MemoryCluster]: Memory clusters
        """
        try:
            if time_window is None:
                time_window = timedelta(days=7)
            
            clusters = await self.vector_memory.cluster_memories(agent_id, time_window)
            
            # Update statistics
            self.stats['clusters_created'] += len(clusters)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Failed to cluster agent memories: {e}")
            return []
    
    async def get_lessons_learned(self, scenario_type: str, role: Optional[Role] = None) -> List[Dict[str, Any]]:
        """
        Get lessons learned for a scenario type and role
        
        Args:
            scenario_type: Type of scenario
            role: Optional role filter
            
        Returns:
            List of lessons learned
        """
        try:
            lessons = await self.knowledge_base.get_lessons_learned(scenario_type)
            
            # Filter by role if specified
            if role:
                role_lessons = [
                    lesson for lesson in lessons
                    if role.value in lesson.applicable_roles
                ]
                return [self._lesson_to_dict(lesson) for lesson in role_lessons]
            
            return [self._lesson_to_dict(lesson) for lesson in lessons]
            
        except Exception as e:
            self.logger.error(f"Failed to get lessons learned: {e}")
            return []
    
    async def update_agent_knowledge(self, agent_id: str, knowledge_update: Dict[str, Any]) -> None:
        """
        Update agent-specific knowledge
        
        Args:
            agent_id: Agent ID
            knowledge_update: Knowledge update data
        """
        try:
            # Store as tactical knowledge in vector memory
            await self.vector_memory.update_tactical_knowledge(agent_id, knowledge_update)
            
            # Update statistics
            self.stats['knowledge_updates'] += 1
            
            self.logger.debug(f"Updated knowledge for agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to update agent knowledge: {e}")
    
    async def search_cross_system(self, query: str, max_results: int = 20) -> Dict[str, Any]:
        """
        Search across both vector memory and knowledge base
        
        Args:
            query: Search query
            max_results: Maximum results per system
            
        Returns:
            Dict containing results from both systems
        """
        try:
            # Search vector memory
            memory_results = await self.retrieve_similar_experiences(
                query_text=query,
                max_results=max_results
            )
            
            # Search knowledge base
            knowledge_results = await self.get_tactical_knowledge(query)
            
            return {
                "experiences": memory_results,
                "knowledge": knowledge_results,
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to search cross-system: {e}")
            return {"error": str(e)}
    
    async def get_agent_memory_profile(self, agent_id: str) -> Dict[str, Any]:
        """
        Get comprehensive memory profile for an agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Dict containing agent memory profile
        """
        try:
            # Get recent experiences
            recent_experiences = await self.retrieve_similar_experiences(
                query_text="",  # Empty query to get all
                agent_id=agent_id,
                max_results=100,
                similarity_threshold=0.0
            )
            
            # Get memory clusters
            clusters = await self.cluster_agent_memories(agent_id)
            
            # Calculate statistics
            successful_experiences = [exp for exp in recent_experiences if exp.metadata.get("success", False)]
            success_rate = len(successful_experiences) / len(recent_experiences) if recent_experiences else 0.0
            
            avg_confidence = sum(exp.metadata.get("confidence", 0.0) for exp in recent_experiences) / len(recent_experiences) if recent_experiences else 0.0
            
            return {
                "agent_id": agent_id,
                "total_experiences": len(recent_experiences),
                "success_rate": success_rate,
                "average_confidence": avg_confidence,
                "memory_clusters": len(clusters),
                "recent_activity": recent_experiences[:10],  # Last 10 experiences
                "cluster_summary": [
                    {
                        "cluster_id": cluster.cluster_id,
                        "name": cluster.cluster_name,
                        "size": len(cluster.experiences),
                        "created": cluster.created_at.isoformat()
                    }
                    for cluster in clusters
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get agent memory profile: {e}")
            return {"error": str(e)}
    
    async def _extract_knowledge_from_experience(self, experience: Experience) -> None:
        """Extract knowledge from experience and update knowledge base"""
        try:
            # Extract lessons learned
            if experience.lessons_learned:
                from .knowledge_base import Lesson
                import uuid
                
                for lesson_text in experience.lessons_learned:
                    lesson = Lesson(
                        lesson_id=str(uuid.uuid4()),
                        scenario_type="general",
                        lesson_text=lesson_text,
                        category="experience",
                        importance_score=experience.confidence_score,
                        applicable_roles=["general"],
                        mitre_techniques=experience.mitre_attack_mapping,
                        created_at=experience.timestamp,
                        validated=False,
                        metadata={
                            "source_experience": experience.experience_id,
                            "agent_id": experience.agent_id
                        }
                    )
                    
                    await self.knowledge_base.add_lesson_learned(lesson)
            
            # Extract TTP information if available
            if hasattr(experience, 'action_taken') and experience.action_taken:
                # This would extract TTP information from the action
                # Implementation depends on action structure
                pass
            
        except Exception as e:
            self.logger.error(f"Failed to extract knowledge from experience: {e}")
    
    def _lesson_to_dict(self, lesson) -> Dict[str, Any]:
        """Convert lesson object to dictionary"""
        return {
            "lesson_id": lesson.lesson_id,
            "scenario_type": lesson.scenario_type,
            "lesson_text": lesson.lesson_text,
            "category": lesson.category,
            "importance_score": lesson.importance_score,
            "applicable_roles": lesson.applicable_roles,
            "mitre_techniques": lesson.mitre_techniques,
            "created_at": lesson.created_at.isoformat(),
            "validated": lesson.validated,
            "metadata": lesson.metadata
        }
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while self.running:
            try:
                await asyncio.sleep(self.config.cleanup_interval.total_seconds())
                
                if not self.running:
                    break
                
                # Clean up old memories
                cleaned_count = await self.vector_memory.cleanup_old_memories(self.config.max_memory_age)
                
                # Update statistics
                self.stats['cleanup_runs'] += 1
                
                self.logger.info(f"Cleanup completed: removed {cleaned_count} old memories")
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _clustering_loop(self) -> None:
        """Background clustering loop"""
        while self.running:
            try:
                await asyncio.sleep(self.config.clustering_interval.total_seconds())
                
                if not self.running:
                    break
                
                # Get all agents with recent activity
                # This would need to be implemented based on how we track active agents
                # For now, skip automatic clustering
                
                self.logger.debug("Clustering loop iteration completed")
                
            except Exception as e:
                self.logger.error(f"Error in clustering loop: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        try:
            vector_stats = self.vector_memory.get_memory_stats()
            knowledge_stats = self.knowledge_base.get_knowledge_base_stats()
            
            return {
                "memory_manager": {
                    "initialized": self.initialized,
                    "running": self.running,
                    "config": {
                        "vector_db_type": self.config.vector_db_type.value,
                        "embedding_model": self.config.embedding_model,
                        "max_memory_age_days": self.config.max_memory_age.days,
                        "clustering_enabled": self.config.clustering_enabled
                    },
                    "statistics": self.stats
                },
                "vector_memory": vector_stats,
                "knowledge_base": knowledge_stats
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory statistics: {e}")
            return {"error": str(e)}
    
    async def shutdown(self) -> None:
        """Shutdown the memory manager"""
        try:
            self.logger.info("Shutting down memory manager")
            self.running = False
            
            # Cancel background tasks
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            if self.clustering_task:
                self.clustering_task.cancel()
                try:
                    await self.clustering_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown subsystems
            await self.vector_memory.shutdown()
            await self.knowledge_base.shutdown()
            
            self.initialized = False
            
            self.logger.info("Memory manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during memory manager shutdown: {e}")

# Factory function
def create_memory_manager(config: MemoryConfig = None) -> MemoryManager:
    """Create a memory manager instance"""
    return MemoryManager(config or MemoryConfig())