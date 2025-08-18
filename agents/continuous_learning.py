#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Continuous Learning System
Integrated continuous learning with human-in-the-loop feedback and knowledge distillation
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

from .base_agent import Experience, Team, Role, ActionResult
from .learning_system import LearningSystem, PerformanceMetrics, LearningConfig
from .human_in_the_loop import HumanInTheLoopInterface, HumanFeedback, ValidationRequest
from .knowledge_distillation import KnowledgeDistillationPipeline, DistillationType, DistillationResult
from memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class LearningPolicyType(Enum):
    """Types of learning policies"""
    CONSERVATIVE = "conservative"      # Slow, careful learning
    AGGRESSIVE = "aggressive"          # Fast, experimental learning
    BALANCED = "balanced"             # Moderate learning approach
    HUMAN_GUIDED = "human_guided"     # Heavy human oversight
    AUTONOMOUS = "autonomous"         # Minimal human intervention

class ModelUpdateStrategy(Enum):
    """Strategies for automated model updates"""
    IMMEDIATE = "immediate"           # Update immediately after feedback
    BATCH = "batch"                  # Update in batches
    SCHEDULED = "scheduled"          # Update on schedule
    THRESHOLD_BASED = "threshold"    # Update when threshold reached
    HUMAN_APPROVED = "human_approved" # Update only with human approval

@dataclass
class LearningPolicy:
    """Policy configuration for continuous learning"""
    policy_id: str
    policy_type: LearningPolicyType
    update_strategy: ModelUpdateStrategy
    
    # Thresholds and parameters
    feedback_threshold: int = 10          # Min feedback before update
    confidence_threshold: float = 0.7     # Min confidence for auto-update
    human_approval_required: bool = False # Require human approval
    distillation_frequency: int = 100     # Distill every N experiences
    
    # Learning rates and parameters
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    memory_retention_rate: float = 0.9
    
    # Quality controls
    min_quality_score: float = 0.6
    max_compression_ratio: float = 0.8
    
    # Timing parameters
    update_interval: timedelta = timedelta(hours=1)
    evaluation_window: timedelta = timedelta(days=1)
    
    created_by: str = "system"
    timestamp: datetime = field(default_factory=datetime.now)
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LearningUpdate:
    """Record of a learning system update"""
    update_id: str
    agent_id: str
    update_type: str
    trigger: str
    
    # Before/after state
    previous_performance: Optional[PerformanceMetrics] = None
    updated_performance: Optional[PerformanceMetrics] = None
    
    # Update details
    feedback_incorporated: List[str] = field(default_factory=list)
    distillation_applied: Optional[str] = None
    model_changes: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    human_approved: bool = False
    validation_results: Dict[str, Any] = field(default_factory=dict)
    
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None

class ContinuousLearningSystem:
    """
    Integrated continuous learning system with human feedback and knowledge distillation
    """
    
    def __init__(self, 
                 learning_system: LearningSystem = None,
                 human_interface: HumanInTheLoopInterface = None,
                 distillation_pipeline: KnowledgeDistillationPipeline = None,
                 memory_manager: MemoryManager = None):
        
        self.learning_system = learning_system
        self.human_interface = human_interface
        self.distillation_pipeline = distillation_pipeline
        self.memory_manager = memory_manager
        
        # Learning policies and updates
        self.learning_policies = {}  # agent_id -> LearningPolicy
        self.update_history = {}     # agent_id -> List[LearningUpdate]
        self.pending_updates = {}    # agent_id -> List[pending updates]
        
        # Feedback integration
        self.feedback_buffer = {}    # agent_id -> List[HumanFeedback]
        self.experience_buffer = {}  # agent_id -> List[Experience]
        
        # Background tasks
        self.learning_tasks = {}     # agent_id -> asyncio.Task
        self.update_scheduler = None
        
        # Configuration
        self.default_policy_type = LearningPolicyType.BALANCED
        self.max_buffer_size = 1000
        self.update_batch_size = 50
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the continuous learning system"""
        try:
            self.logger.info("Initializing continuous learning system")
            
            # Initialize components if not provided
            if self.learning_system is None:
                from .learning_system import LearningSystem, LearningConfig
                self.learning_system = LearningSystem(LearningConfig())
                await self.learning_system.initialize()
            
            if self.human_interface is None:
                from .human_in_the_loop import create_human_in_the_loop_interface
                self.human_interface = create_human_in_the_loop_interface(self.learning_system)
                await self.human_interface.initialize()
            
            if self.distillation_pipeline is None:
                from .knowledge_distillation import create_knowledge_distillation_pipeline
                self.distillation_pipeline = create_knowledge_distillation_pipeline()
                await self.distillation_pipeline.initialize()
            
            if self.memory_manager is None:
                from memory.memory_manager import create_memory_manager
                self.memory_manager = create_memory_manager()
                await self.memory_manager.initialize()
            
            # Start background tasks
            self.update_scheduler = asyncio.create_task(self._run_update_scheduler())
            
            # Register human feedback callback
            self.human_interface.register_validation_callback(
                "continuous_learning",
                self._handle_validation_feedback
            )
            
            self.initialized = True
            self.logger.info("Continuous learning system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize continuous learning system: {e}")
            raise
    
    async def register_agent(self, 
                           agent_id: str,
                           policy_type: LearningPolicyType = None,
                           custom_policy: LearningPolicy = None) -> None:
        """
        Register an agent for continuous learning
        
        Args:
            agent_id: ID of the agent to register
            policy_type: Type of learning policy to use
            custom_policy: Custom learning policy (overrides policy_type)
        """
        try:
            if custom_policy:
                policy = custom_policy
            else:
                policy_type = policy_type or self.default_policy_type
                policy = self._create_default_policy(agent_id, policy_type)
            
            self.learning_policies[agent_id] = policy
            self.feedback_buffer[agent_id] = []
            self.experience_buffer[agent_id] = []
            self.update_history[agent_id] = []
            self.pending_updates[agent_id] = []
            
            # Start learning task for agent
            self.learning_tasks[agent_id] = asyncio.create_task(
                self._run_agent_learning_loop(agent_id)
            )
            
            self.logger.info(f"Registered agent {agent_id} for continuous learning with {policy.policy_type.value} policy")
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent_id}: {e}")
            raise
    
    async def add_experience(self, agent_id: str, experience: Experience) -> None:
        """Add an experience for continuous learning"""
        try:
            if agent_id not in self.experience_buffer:
                await self.register_agent(agent_id)
            
            # Add to experience buffer
            self.experience_buffer[agent_id].append(experience)
            
            # Maintain buffer size
            if len(self.experience_buffer[agent_id]) > self.max_buffer_size:
                self.experience_buffer[agent_id] = self.experience_buffer[agent_id][-self.max_buffer_size:]
            
            # Add to learning system replay buffer
            await self.learning_system.add_experience_to_replay_buffer(experience)
            
            # Check if distillation is needed
            policy = self.learning_policies[agent_id]
            if len(self.experience_buffer[agent_id]) % policy.distillation_frequency == 0:
                await self._trigger_knowledge_distillation(agent_id)
            
        except Exception as e:
            self.logger.error(f"Failed to add experience for agent {agent_id}: {e}")
    
    async def add_human_feedback(self, agent_id: str, feedback: HumanFeedback) -> None:
        """Add human feedback for continuous learning"""
        try:
            if agent_id not in self.feedback_buffer:
                await self.register_agent(agent_id)
            
            # Add to feedback buffer
            self.feedback_buffer[agent_id].append(feedback)
            
            # Check if update is triggered
            policy = self.learning_policies[agent_id]
            if len(self.feedback_buffer[agent_id]) >= policy.feedback_threshold:
                await self._trigger_learning_update(agent_id, "feedback_threshold")
            
        except Exception as e:
            self.logger.error(f"Failed to add human feedback for agent {agent_id}: {e}")
    
    async def request_learning_update(self, 
                                    agent_id: str,
                                    trigger: str = "manual",
                                    immediate: bool = False) -> str:
        """
        Request a learning update for an agent
        
        Args:
            agent_id: ID of the agent
            trigger: Reason for the update
            immediate: Whether to process immediately
            
        Returns:
            str: Update ID
        """
        try:
            update_id = str(uuid.uuid4())
            
            if immediate:
                result = await self._process_learning_update(agent_id, trigger, update_id)
                return result.update_id
            else:
                # Add to pending updates
                self.pending_updates[agent_id].append({
                    "update_id": update_id,
                    "trigger": trigger,
                    "timestamp": datetime.now()
                })
                return update_id
            
        except Exception as e:
            self.logger.error(f"Failed to request learning update for agent {agent_id}: {e}")
            raise
    
    async def _run_agent_learning_loop(self, agent_id: str) -> None:
        """Run continuous learning loop for an agent"""
        try:
            policy = self.learning_policies[agent_id]
            
            while policy.active:
                try:
                    # Check for scheduled updates
                    if policy.update_strategy == ModelUpdateStrategy.SCHEDULED:
                        last_update = self._get_last_update_time(agent_id)
                        if datetime.now() - last_update >= policy.update_interval:
                            await self._trigger_learning_update(agent_id, "scheduled")
                    
                    # Process pending updates
                    if self.pending_updates[agent_id]:
                        pending = self.pending_updates[agent_id].pop(0)
                        await self._process_learning_update(
                            agent_id, 
                            pending["trigger"], 
                            pending["update_id"]
                        )
                    
                    # Sleep before next iteration
                    await asyncio.sleep(60)  # Check every minute
                    
                except Exception as e:
                    self.logger.error(f"Error in learning loop for agent {agent_id}: {e}")
                    await asyncio.sleep(60)
            
        except Exception as e:
            self.logger.error(f"Learning loop failed for agent {agent_id}: {e}")
    
    async def _run_update_scheduler(self) -> None:
        """Run the global update scheduler"""
        try:
            while self.initialized:
                try:
                    # Check all agents for threshold-based updates
                    for agent_id, policy in self.learning_policies.items():
                        if policy.update_strategy == ModelUpdateStrategy.THRESHOLD_BASED:
                            await self._check_threshold_updates(agent_id)
                    
                    # Sleep before next check
                    await asyncio.sleep(300)  # Check every 5 minutes
                    
                except Exception as e:
                    self.logger.error(f"Error in update scheduler: {e}")
                    await asyncio.sleep(300)
                    
        except Exception as e:
            self.logger.error(f"Update scheduler failed: {e}")
    
    async def _check_threshold_updates(self, agent_id: str) -> None:
        """Check if agent meets thresholds for learning update"""
        try:
            policy = self.learning_policies[agent_id]
            
            # Check feedback threshold
            if len(self.feedback_buffer[agent_id]) >= policy.feedback_threshold:
                await self._trigger_learning_update(agent_id, "feedback_threshold")
                return
            
            # Check experience threshold
            if len(self.experience_buffer[agent_id]) >= policy.distillation_frequency:
                await self._trigger_knowledge_distillation(agent_id)
            
            # Check performance degradation
            current_performance = await self.learning_system.get_agent_performance_history(agent_id)
            if current_performance and len(current_performance) >= 2:
                recent = current_performance[-1]
                previous = current_performance[-2]
                
                if recent.success_rate < previous.success_rate - 0.1:  # 10% degradation
                    await self._trigger_learning_update(agent_id, "performance_degradation")
            
        except Exception as e:
            self.logger.error(f"Failed to check thresholds for agent {agent_id}: {e}")
    
    async def _trigger_learning_update(self, agent_id: str, trigger: str) -> None:
        """Trigger a learning update for an agent"""
        try:
            update_id = str(uuid.uuid4())
            
            policy = self.learning_policies[agent_id]
            
            # Check if human approval is required
            if policy.human_approval_required or policy.policy_type == LearningPolicyType.HUMAN_GUIDED:
                # Request human validation
                await self._request_update_validation(agent_id, update_id, trigger)
            else:
                # Process update directly
                await self._process_learning_update(agent_id, trigger, update_id)
            
        except Exception as e:
            self.logger.error(f"Failed to trigger learning update for agent {agent_id}: {e}")
    
    async def _trigger_knowledge_distillation(self, agent_id: str) -> None:
        """Trigger knowledge distillation for an agent"""
        try:
            experiences = self.experience_buffer[agent_id]
            feedback = self.feedback_buffer[agent_id]
            
            if not experiences:
                return
            
            # Perform distillation
            result = await self.distillation_pipeline.distill_agent_knowledge(
                agent_id=agent_id,
                experiences=experiences,
                distillation_type=DistillationType.BEHAVIOR_PRUNING,
                human_feedback=feedback
            )
            
            # Apply distillation results
            if result.quality_score >= self.learning_policies[agent_id].min_quality_score:
                await self._apply_distillation_results(agent_id, result)
            
            self.logger.info(f"Completed knowledge distillation for agent {agent_id}: {result.compression_ratio:.2f} compression")
            
        except Exception as e:
            self.logger.error(f"Failed to trigger knowledge distillation for agent {agent_id}: {e}")
    
    async def _process_learning_update(self, agent_id: str, trigger: str, update_id: str) -> LearningUpdate:
        """Process a learning update for an agent"""
        try:
            policy = self.learning_policies[agent_id]
            
            # Get current performance
            previous_performance = None
            performance_history = await self.learning_system.get_agent_performance_history(agent_id)
            if performance_history:
                previous_performance = performance_history[-1]
            
            # Create update record
            update = LearningUpdate(
                update_id=update_id,
                agent_id=agent_id,
                update_type="continuous_learning",
                trigger=trigger,
                previous_performance=previous_performance
            )
            
            # Process feedback
            feedback_ids = []
            if self.feedback_buffer[agent_id]:
                feedback_ids = await self._process_feedback_batch(agent_id)
                update.feedback_incorporated = feedback_ids
            
            # Apply model changes based on policy
            model_changes = await self._apply_policy_updates(agent_id, policy)
            update.model_changes = model_changes
            
            # Validate update
            validation_results = await self._validate_learning_update(agent_id, update)
            update.validation_results = validation_results
            update.success = validation_results.get("success", False)
            
            if not update.success:
                update.error_message = validation_results.get("error", "Unknown error")
            
            # Get updated performance
            updated_performance = await self.learning_system._calculate_agent_performance(agent_id)
            update.updated_performance = updated_performance
            
            # Store update
            self.update_history[agent_id].append(update)
            
            self.logger.info(f"Processed learning update for agent {agent_id}: {trigger}")
            return update
            
        except Exception as e:
            self.logger.error(f"Failed to process learning update for agent {agent_id}: {e}")
            
            # Create error update record
            error_update = LearningUpdate(
                update_id=update_id,
                agent_id=agent_id,
                update_type="continuous_learning",
                trigger=trigger,
                success=False,
                error_message=str(e)
            )
            
            self.update_history[agent_id].append(error_update)
            return error_update
    
    async def _process_feedback_batch(self, agent_id: str) -> List[str]:
        """Process a batch of human feedback"""
        try:
            feedback_batch = self.feedback_buffer[agent_id].copy()
            self.feedback_buffer[agent_id].clear()
            
            feedback_ids = []
            
            for feedback in feedback_batch:
                # Process different types of feedback
                if feedback.feedback_type.value == "performance_rating":
                    await self._apply_performance_feedback(agent_id, feedback)
                elif feedback.feedback_type.value == "behavior_tagging":
                    await self._apply_behavior_feedback(agent_id, feedback)
                elif feedback.feedback_type.value == "strategy_correction":
                    await self._apply_strategy_feedback(agent_id, feedback)
                
                feedback_ids.append(feedback.feedback_id)
            
            return feedback_ids
            
        except Exception as e:
            self.logger.error(f"Failed to process feedback batch for agent {agent_id}: {e}")
            return []
    
    async def _apply_performance_feedback(self, agent_id: str, feedback: HumanFeedback) -> None:
        """Apply performance rating feedback"""
        try:
            if feedback.performance_rating is not None:
                # Adjust learning parameters based on performance rating
                policy = self.learning_policies[agent_id]
                
                if feedback.performance_rating < 0.5:
                    # Poor performance - increase exploration
                    policy.exploration_rate = min(0.5, policy.exploration_rate * 1.2)
                elif feedback.performance_rating > 0.8:
                    # Good performance - reduce exploration
                    policy.exploration_rate = max(0.05, policy.exploration_rate * 0.9)
            
        except Exception as e:
            self.logger.error(f"Failed to apply performance feedback: {e}")
    
    async def _apply_behavior_feedback(self, agent_id: str, feedback: HumanFeedback) -> None:
        """Apply behavior tagging feedback"""
        try:
            if feedback.behavior_tags and feedback.correctness_score is not None:
                # Store behavior tags in memory for future reference
                if self.memory_manager:
                    # This would update the agent's behavior model
                    pass
            
        except Exception as e:
            self.logger.error(f"Failed to apply behavior feedback: {e}")
    
    async def _apply_strategy_feedback(self, agent_id: str, feedback: HumanFeedback) -> None:
        """Apply strategy correction feedback"""
        try:
            if feedback.strategy_modifications:
                # Apply strategy modifications to the learning system
                # This would update the agent's strategy parameters
                pass
            
        except Exception as e:
            self.logger.error(f"Failed to apply strategy feedback: {e}")
    
    async def _apply_policy_updates(self, agent_id: str, policy: LearningPolicy) -> Dict[str, Any]:
        """Apply policy-based updates to the agent"""
        try:
            changes = {}
            
            # Update learning rate based on policy
            if policy.policy_type == LearningPolicyType.AGGRESSIVE:
                changes["learning_rate"] = policy.learning_rate * 1.5
            elif policy.policy_type == LearningPolicyType.CONSERVATIVE:
                changes["learning_rate"] = policy.learning_rate * 0.7
            else:
                changes["learning_rate"] = policy.learning_rate
            
            # Update exploration rate
            changes["exploration_rate"] = policy.exploration_rate
            
            # Update memory retention
            changes["memory_retention_rate"] = policy.memory_retention_rate
            
            return changes
            
        except Exception as e:
            self.logger.error(f"Failed to apply policy updates: {e}")
            return {}
    
    async def _validate_learning_update(self, agent_id: str, update: LearningUpdate) -> Dict[str, Any]:
        """Validate a learning update"""
        try:
            validation_results = {
                "success": True,
                "checks_passed": [],
                "checks_failed": [],
                "warnings": []
            }
            
            # Check if performance improved or maintained
            if update.previous_performance and update.updated_performance:
                prev_success = update.previous_performance.success_rate
                new_success = update.updated_performance.success_rate
                
                if new_success >= prev_success - 0.05:  # Allow 5% degradation
                    validation_results["checks_passed"].append("performance_maintained")
                else:
                    validation_results["checks_failed"].append("performance_degraded")
                    validation_results["success"] = False
            
            # Check if model changes are reasonable
            if update.model_changes:
                for param, value in update.model_changes.items():
                    if isinstance(value, (int, float)):
                        if 0 <= value <= 1:  # Reasonable range for most parameters
                            validation_results["checks_passed"].append(f"valid_{param}")
                        else:
                            validation_results["warnings"].append(f"unusual_{param}_value")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Failed to validate learning update: {e}")
            return {"success": False, "error": str(e)}
    
    async def _apply_distillation_results(self, agent_id: str, result: DistillationResult) -> None:
        """Apply knowledge distillation results"""
        try:
            # Update experience buffer with filtered experiences
            if result.distillation_type == DistillationType.BEHAVIOR_PRUNING:
                # Remove pruned behaviors from future consideration
                pruned_set = set(result.pruned_behaviors)
                
                # Filter experience buffer
                filtered_experiences = []
                for exp in self.experience_buffer[agent_id]:
                    action_str = str(exp.action_taken) if exp.action_taken else "unknown"
                    if action_str not in pruned_set:
                        filtered_experiences.append(exp)
                
                self.experience_buffer[agent_id] = filtered_experiences
            
            # Store patterns in memory
            if result.extracted_patterns and self.memory_manager:
                for pattern in result.extracted_patterns:
                    # Store pattern as structured knowledge
                    pass
            
            self.logger.info(f"Applied distillation results for agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply distillation results: {e}")
    
    async def _request_update_validation(self, agent_id: str, update_id: str, trigger: str) -> None:
        """Request human validation for a learning update"""
        try:
            # Create a mock action for validation
            from .base_agent import Action
            
            update_action = Action(
                action_id=update_id,
                action_type="learning_update",
                parameters={"trigger": trigger, "agent_id": agent_id},
                timestamp=datetime.now()
            )
            
            # Request validation
            validation_id = await self.human_interface.request_action_validation(
                agent_id=agent_id,
                action=update_action,
                context={"update_type": "continuous_learning", "trigger": trigger},
                reasoning=f"Requesting approval for learning update triggered by {trigger}",
                confidence_score=0.8
            )
            
            self.logger.info(f"Requested validation for learning update {update_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to request update validation: {e}")
    
    async def _handle_validation_feedback(self, request: ValidationRequest) -> None:
        """Handle validation feedback from human interface"""
        try:
            # Check if this is a learning update validation
            if (hasattr(request.action, 'action_type') and 
                request.action.action_type == "learning_update"):
                
                # Process based on validation status
                if request.status.value == "approved":
                    # Process the learning update
                    await self._process_learning_update(
                        request.agent_id,
                        request.context.get("trigger", "human_approved"),
                        request.action.action_id
                    )
                elif request.status.value == "rejected":
                    # Log rejection and don't process update
                    self.logger.info(f"Learning update rejected for agent {request.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle validation feedback: {e}")
    
    def _create_default_policy(self, agent_id: str, policy_type: LearningPolicyType) -> LearningPolicy:
        """Create a default learning policy"""
        base_policy = LearningPolicy(
            policy_id=str(uuid.uuid4()),
            policy_type=policy_type,
            update_strategy=ModelUpdateStrategy.THRESHOLD_BASED
        )
        
        # Customize based on policy type
        if policy_type == LearningPolicyType.CONSERVATIVE:
            base_policy.learning_rate = 0.005
            base_policy.exploration_rate = 0.05
            base_policy.feedback_threshold = 20
            base_policy.human_approval_required = True
            
        elif policy_type == LearningPolicyType.AGGRESSIVE:
            base_policy.learning_rate = 0.02
            base_policy.exploration_rate = 0.2
            base_policy.feedback_threshold = 5
            base_policy.update_strategy = ModelUpdateStrategy.IMMEDIATE
            
        elif policy_type == LearningPolicyType.HUMAN_GUIDED:
            base_policy.human_approval_required = True
            base_policy.update_strategy = ModelUpdateStrategy.HUMAN_APPROVED
            base_policy.feedback_threshold = 1
            
        elif policy_type == LearningPolicyType.AUTONOMOUS:
            base_policy.human_approval_required = False
            base_policy.update_strategy = ModelUpdateStrategy.BATCH
            base_policy.feedback_threshold = 15
        
        return base_policy
    
    def _get_last_update_time(self, agent_id: str) -> datetime:
        """Get the timestamp of the last update for an agent"""
        if agent_id in self.update_history and self.update_history[agent_id]:
            return self.update_history[agent_id][-1].timestamp
        else:
            return datetime.now() - timedelta(days=1)  # Default to 1 day ago
    
    # Query methods
    async def get_learning_policy(self, agent_id: str) -> Optional[LearningPolicy]:
        """Get the learning policy for an agent"""
        return self.learning_policies.get(agent_id)
    
    async def update_learning_policy(self, agent_id: str, policy: LearningPolicy) -> None:
        """Update the learning policy for an agent"""
        self.learning_policies[agent_id] = policy
        self.logger.info(f"Updated learning policy for agent {agent_id}")
    
    async def get_update_history(self, agent_id: str) -> List[LearningUpdate]:
        """Get the update history for an agent"""
        return self.update_history.get(agent_id, [])
    
    async def get_system_statistics(self) -> Dict[str, Any]:
        """Get statistics about the continuous learning system"""
        total_updates = sum(len(history) for history in self.update_history.values())
        successful_updates = sum(
            len([u for u in history if u.success]) 
            for history in self.update_history.values()
        )
        
        return {
            "registered_agents": len(self.learning_policies),
            "total_updates": total_updates,
            "successful_updates": successful_updates,
            "success_rate": successful_updates / total_updates if total_updates > 0 else 0.0,
            "active_learning_tasks": len([t for t in self.learning_tasks.values() if not t.done()]),
            "total_feedback": sum(len(buffer) for buffer in self.feedback_buffer.values()),
            "total_experiences": sum(len(buffer) for buffer in self.experience_buffer.values())
        }
    
    async def shutdown(self) -> None:
        """Shutdown the continuous learning system"""
        try:
            self.logger.info("Shutting down continuous learning system")
            
            # Cancel all learning tasks
            for task in self.learning_tasks.values():
                task.cancel()
            
            # Cancel update scheduler
            if self.update_scheduler:
                self.update_scheduler.cancel()
            
            # Shutdown components
            if self.human_interface:
                await self.human_interface.shutdown()
            
            self.initialized = False
            self.logger.info("Continuous learning system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Factory function
def create_continuous_learning_system(
    learning_system: LearningSystem = None,
    human_interface: HumanInTheLoopInterface = None,
    distillation_pipeline: KnowledgeDistillationPipeline = None,
    memory_manager: MemoryManager = None
) -> ContinuousLearningSystem:
    """Create and return a continuous learning system"""
    return ContinuousLearningSystem(
        learning_system=learning_system,
        human_interface=human_interface,
        distillation_pipeline=distillation_pipeline,
        memory_manager=memory_manager
    )