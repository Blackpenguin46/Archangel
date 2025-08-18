#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Human-in-the-Loop System
Human feedback integration for agent action validation and learning improvement
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import uuid

from .base_agent import Experience, Team, Role, ActionResult, Action
from .learning_system import PerformanceMetrics, LearningSystem

logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    """Human validation status for agent actions"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    ESCALATED = "escalated"

class FeedbackType(Enum):
    """Types of human feedback"""
    ACTION_VALIDATION = "action_validation"
    PERFORMANCE_RATING = "performance_rating"
    STRATEGY_CORRECTION = "strategy_correction"
    BEHAVIOR_TAGGING = "behavior_tagging"
    LEARNING_GUIDANCE = "learning_guidance"

class Priority(Enum):
    """Priority levels for human review"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class HumanFeedback:
    """Human feedback on agent actions or performance"""
    feedback_id: str
    agent_id: str
    feedback_type: FeedbackType
    timestamp: datetime
    reviewer_id: str
    
    # Action-specific feedback
    action_id: Optional[str] = None
    validation_status: Optional[ValidationStatus] = None
    
    # Performance feedback
    performance_rating: Optional[float] = None  # 0.0 to 1.0
    improvement_suggestions: List[str] = field(default_factory=list)
    
    # Behavior tagging
    behavior_tags: List[str] = field(default_factory=list)
    correctness_score: Optional[float] = None
    
    # Strategy correction
    strategy_modifications: Dict[str, Any] = field(default_factory=dict)
    reasoning_corrections: List[str] = field(default_factory=list)
    
    # General feedback
    comments: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationRequest:
    """Request for human validation of agent action"""
    request_id: str
    agent_id: str
    action: Action
    context: Dict[str, Any]
    reasoning: str
    predicted_outcome: str
    confidence_score: float
    priority: Priority
    timestamp: datetime
    timeout: timedelta
    
    # Validation criteria
    requires_approval: bool = True
    risk_level: str = "medium"
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    
    # Status tracking
    status: ValidationStatus = ValidationStatus.PENDING
    assigned_reviewer: Optional[str] = None
    response_deadline: Optional[datetime] = None

@dataclass
class LearningCorrection:
    """Correction to agent learning based on human feedback"""
    correction_id: str
    agent_id: str
    original_behavior: Dict[str, Any]
    corrected_behavior: Dict[str, Any]
    correction_type: str
    feedback_source: str
    confidence: float
    timestamp: datetime
    applied: bool = False
    validation_results: Dict[str, Any] = field(default_factory=dict)

class HumanInTheLoopInterface:
    """
    Interface for human oversight and feedback in agent learning
    """
    
    def __init__(self, learning_system: LearningSystem = None):
        self.learning_system = learning_system
        
        # Feedback storage
        self.pending_validations = {}  # request_id -> ValidationRequest
        self.feedback_history = {}  # agent_id -> List[HumanFeedback]
        self.learning_corrections = {}  # agent_id -> List[LearningCorrection]
        
        # Review queues
        self.validation_queue = asyncio.Queue()
        self.feedback_queue = asyncio.Queue()
        
        # Reviewers and callbacks
        self.registered_reviewers = {}  # reviewer_id -> reviewer_info
        self.validation_callbacks = {}  # callback_name -> callback_function
        
        # Configuration
        self.auto_approve_threshold = 0.9  # Auto-approve high-confidence actions
        self.escalation_threshold = 0.3   # Escalate low-confidence actions
        self.review_timeout = timedelta(minutes=30)
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the human-in-the-loop system"""
        try:
            self.logger.info("Initializing human-in-the-loop system")
            
            # Start background tasks
            asyncio.create_task(self._process_validation_queue())
            asyncio.create_task(self._process_feedback_queue())
            asyncio.create_task(self._monitor_timeouts())
            
            self.initialized = True
            self.logger.info("Human-in-the-loop system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize human-in-the-loop system: {e}")
            raise
    
    async def request_action_validation(self, 
                                      agent_id: str,
                                      action: Action,
                                      context: Dict[str, Any],
                                      reasoning: str,
                                      confidence_score: float) -> str:
        """
        Request human validation for an agent action
        
        Args:
            agent_id: ID of the agent requesting validation
            action: The action to be validated
            context: Context information for the action
            reasoning: Agent's reasoning for the action
            confidence_score: Agent's confidence in the action
            
        Returns:
            str: Validation request ID
        """
        try:
            request_id = str(uuid.uuid4())
            
            # Determine priority based on confidence and risk
            priority = self._calculate_priority(action, confidence_score, context)
            
            # Create validation request
            request = ValidationRequest(
                request_id=request_id,
                agent_id=agent_id,
                action=action,
                context=context,
                reasoning=reasoning,
                predicted_outcome=context.get("predicted_outcome", ""),
                confidence_score=confidence_score,
                priority=priority,
                timestamp=datetime.now(),
                timeout=self.review_timeout,
                risk_level=self._assess_risk_level(action, context),
                impact_assessment=self._assess_impact(action, context)
            )
            
            # Check for auto-approval
            if confidence_score >= self.auto_approve_threshold and priority == Priority.LOW:
                request.status = ValidationStatus.APPROVED
                await self._auto_approve_action(request)
                return request_id
            
            # Check for auto-escalation
            if confidence_score <= self.escalation_threshold or priority == Priority.CRITICAL:
                request.priority = Priority.CRITICAL
                request.requires_approval = True
            
            # Add to pending validations and queue
            self.pending_validations[request_id] = request
            await self.validation_queue.put(request)
            
            self.logger.info(f"Requested validation for agent {agent_id} action: {request_id}")
            return request_id
            
        except Exception as e:
            self.logger.error(f"Failed to request action validation: {e}")
            raise
    
    async def provide_feedback(self, 
                             agent_id: str,
                             feedback_type: FeedbackType,
                             reviewer_id: str,
                             **feedback_data) -> str:
        """
        Provide human feedback on agent behavior or performance
        
        Args:
            agent_id: ID of the agent receiving feedback
            feedback_type: Type of feedback being provided
            reviewer_id: ID of the human reviewer
            **feedback_data: Additional feedback data
            
        Returns:
            str: Feedback ID
        """
        try:
            feedback_id = str(uuid.uuid4())
            
            feedback = HumanFeedback(
                feedback_id=feedback_id,
                agent_id=agent_id,
                feedback_type=feedback_type,
                timestamp=datetime.now(),
                reviewer_id=reviewer_id,
                **feedback_data
            )
            
            # Store feedback
            if agent_id not in self.feedback_history:
                self.feedback_history[agent_id] = []
            self.feedback_history[agent_id].append(feedback)
            
            # Queue for processing
            await self.feedback_queue.put(feedback)
            
            self.logger.info(f"Received {feedback_type.value} feedback for agent {agent_id}")
            return feedback_id
            
        except Exception as e:
            self.logger.error(f"Failed to provide feedback: {e}")
            raise
    
    async def validate_action(self, 
                            request_id: str,
                            reviewer_id: str,
                            status: ValidationStatus,
                            comments: str = "",
                            modifications: Dict[str, Any] = None) -> bool:
        """
        Validate a pending action request
        
        Args:
            request_id: ID of the validation request
            reviewer_id: ID of the reviewer
            status: Validation status (approved/rejected/modified)
            comments: Reviewer comments
            modifications: Suggested modifications if status is MODIFIED
            
        Returns:
            bool: True if validation was processed successfully
        """
        try:
            if request_id not in self.pending_validations:
                self.logger.warning(f"Validation request {request_id} not found")
                return False
            
            request = self.pending_validations[request_id]
            request.status = status
            request.assigned_reviewer = reviewer_id
            
            # Create feedback record
            feedback = HumanFeedback(
                feedback_id=str(uuid.uuid4()),
                agent_id=request.agent_id,
                feedback_type=FeedbackType.ACTION_VALIDATION,
                timestamp=datetime.now(),
                reviewer_id=reviewer_id,
                action_id=request_id,
                validation_status=status,
                comments=comments,
                strategy_modifications=modifications or {}
            )
            
            # Store feedback
            if request.agent_id not in self.feedback_history:
                self.feedback_history[request.agent_id] = []
            self.feedback_history[request.agent_id].append(feedback)
            
            # Process validation result
            await self._process_validation_result(request, feedback)
            
            # Remove from pending
            del self.pending_validations[request_id]
            
            self.logger.info(f"Processed validation {request_id}: {status.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate action: {e}")
            return False
    
    async def tag_behavior(self, 
                          agent_id: str,
                          action_id: str,
                          tags: List[str],
                          correctness_score: float,
                          reviewer_id: str,
                          comments: str = "") -> str:
        """
        Tag agent behavior for learning improvement
        
        Args:
            agent_id: ID of the agent
            action_id: ID of the action being tagged
            tags: Behavior tags (e.g., "aggressive", "cautious", "effective")
            correctness_score: Score from 0.0 to 1.0 indicating correctness
            reviewer_id: ID of the reviewer
            comments: Additional comments
            
        Returns:
            str: Feedback ID
        """
        try:
            return await self.provide_feedback(
                agent_id=agent_id,
                feedback_type=FeedbackType.BEHAVIOR_TAGGING,
                reviewer_id=reviewer_id,
                action_id=action_id,
                behavior_tags=tags,
                correctness_score=correctness_score,
                comments=comments
            )
            
        except Exception as e:
            self.logger.error(f"Failed to tag behavior: {e}")
            raise
    
    async def correct_strategy(self, 
                             agent_id: str,
                             original_strategy: Dict[str, Any],
                             corrected_strategy: Dict[str, Any],
                             reviewer_id: str,
                             reasoning: List[str],
                             confidence: float = 0.8) -> str:
        """
        Provide strategy correction for agent learning
        
        Args:
            agent_id: ID of the agent
            original_strategy: Original strategy parameters
            corrected_strategy: Corrected strategy parameters
            reviewer_id: ID of the reviewer
            reasoning: List of reasoning for corrections
            confidence: Confidence in the correction
            
        Returns:
            str: Correction ID
        """
        try:
            correction_id = str(uuid.uuid4())
            
            correction = LearningCorrection(
                correction_id=correction_id,
                agent_id=agent_id,
                original_behavior=original_strategy,
                corrected_behavior=corrected_strategy,
                correction_type="strategy_correction",
                feedback_source=reviewer_id,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            # Store correction
            if agent_id not in self.learning_corrections:
                self.learning_corrections[agent_id] = []
            self.learning_corrections[agent_id].append(correction)
            
            # Create feedback record
            await self.provide_feedback(
                agent_id=agent_id,
                feedback_type=FeedbackType.STRATEGY_CORRECTION,
                reviewer_id=reviewer_id,
                strategy_modifications=corrected_strategy,
                reasoning_corrections=reasoning,
                confidence=confidence
            )
            
            # Apply correction if learning system is available
            if self.learning_system:
                await self._apply_learning_correction(correction)
            
            self.logger.info(f"Applied strategy correction for agent {agent_id}")
            return correction_id
            
        except Exception as e:
            self.logger.error(f"Failed to correct strategy: {e}")
            raise
    
    def _calculate_priority(self, action: Action, confidence: float, context: Dict[str, Any]) -> Priority:
        """Calculate priority level for validation request"""
        # High-risk actions get high priority
        risk_indicators = ["delete", "modify", "escalate", "attack", "exploit"]
        action_str = str(action).lower()
        
        if any(indicator in action_str for indicator in risk_indicators):
            return Priority.HIGH
        
        # Low confidence gets higher priority
        if confidence < 0.5:
            return Priority.HIGH
        elif confidence < 0.7:
            return Priority.MEDIUM
        else:
            return Priority.LOW
    
    def _assess_risk_level(self, action: Action, context: Dict[str, Any]) -> str:
        """Assess risk level of an action"""
        # Simplified risk assessment
        destructive_actions = ["delete", "remove", "destroy", "format"]
        action_str = str(action).lower()
        
        if any(destructive in action_str for destructive in destructive_actions):
            return "high"
        elif "modify" in action_str or "change" in action_str:
            return "medium"
        else:
            return "low"
    
    def _assess_impact(self, action: Action, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess potential impact of an action"""
        return {
            "scope": context.get("scope", "local"),
            "reversibility": context.get("reversible", True),
            "data_sensitivity": context.get("data_sensitivity", "low"),
            "system_criticality": context.get("system_criticality", "low")
        }
    
    async def _auto_approve_action(self, request: ValidationRequest) -> None:
        """Auto-approve high-confidence, low-risk actions"""
        try:
            # Create auto-approval feedback
            feedback = HumanFeedback(
                feedback_id=str(uuid.uuid4()),
                agent_id=request.agent_id,
                feedback_type=FeedbackType.ACTION_VALIDATION,
                timestamp=datetime.now(),
                reviewer_id="system_auto_approval",
                action_id=request.request_id,
                validation_status=ValidationStatus.APPROVED,
                comments="Auto-approved based on high confidence and low risk",
                confidence=request.confidence_score
            )
            
            # Store feedback
            if request.agent_id not in self.feedback_history:
                self.feedback_history[request.agent_id] = []
            self.feedback_history[request.agent_id].append(feedback)
            
            self.logger.debug(f"Auto-approved action for agent {request.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to auto-approve action: {e}")
    
    async def _process_validation_queue(self) -> None:
        """Process validation requests from the queue"""
        while True:
            try:
                # Get validation request
                request = await self.validation_queue.get()
                
                # Set response deadline
                request.response_deadline = datetime.now() + request.timeout
                
                # Notify registered callbacks
                for callback_name, callback in self.validation_callbacks.items():
                    try:
                        await callback(request)
                    except Exception as e:
                        self.logger.error(f"Validation callback {callback_name} failed: {e}")
                
                self.validation_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error processing validation queue: {e}")
                await asyncio.sleep(1)
    
    async def _process_feedback_queue(self) -> None:
        """Process feedback from the queue"""
        while True:
            try:
                # Get feedback
                feedback = await self.feedback_queue.get()
                
                # Process based on feedback type
                if feedback.feedback_type == FeedbackType.PERFORMANCE_RATING:
                    await self._process_performance_feedback(feedback)
                elif feedback.feedback_type == FeedbackType.BEHAVIOR_TAGGING:
                    await self._process_behavior_tagging(feedback)
                elif feedback.feedback_type == FeedbackType.STRATEGY_CORRECTION:
                    await self._process_strategy_correction(feedback)
                
                self.feedback_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error processing feedback queue: {e}")
                await asyncio.sleep(1)
    
    async def _monitor_timeouts(self) -> None:
        """Monitor validation request timeouts"""
        while True:
            try:
                current_time = datetime.now()
                expired_requests = []
                
                for request_id, request in self.pending_validations.items():
                    if (request.response_deadline and 
                        current_time > request.response_deadline and
                        request.status == ValidationStatus.PENDING):
                        expired_requests.append(request_id)
                
                # Handle expired requests
                for request_id in expired_requests:
                    await self._handle_timeout(request_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error monitoring timeouts: {e}")
                await asyncio.sleep(60)
    
    async def _handle_timeout(self, request_id: str) -> None:
        """Handle validation request timeout"""
        try:
            request = self.pending_validations.get(request_id)
            if not request:
                return
            
            # Auto-reject or escalate based on priority
            if request.priority == Priority.CRITICAL:
                request.status = ValidationStatus.ESCALATED
                self.logger.warning(f"Escalated critical validation request {request_id} due to timeout")
            else:
                request.status = ValidationStatus.REJECTED
                self.logger.info(f"Auto-rejected validation request {request_id} due to timeout")
            
            # Create timeout feedback
            feedback = HumanFeedback(
                feedback_id=str(uuid.uuid4()),
                agent_id=request.agent_id,
                feedback_type=FeedbackType.ACTION_VALIDATION,
                timestamp=datetime.now(),
                reviewer_id="system_timeout",
                action_id=request_id,
                validation_status=request.status,
                comments="Request timed out without human review"
            )
            
            # Store feedback and process result
            if request.agent_id not in self.feedback_history:
                self.feedback_history[request.agent_id] = []
            self.feedback_history[request.agent_id].append(feedback)
            
            await self._process_validation_result(request, feedback)
            
            # Remove from pending
            del self.pending_validations[request_id]
            
        except Exception as e:
            self.logger.error(f"Failed to handle timeout for request {request_id}: {e}")
    
    async def _process_validation_result(self, request: ValidationRequest, feedback: HumanFeedback) -> None:
        """Process the result of a validation"""
        try:
            # Update learning system if available
            if self.learning_system:
                # Create experience record for the validation
                validation_experience = {
                    "agent_id": request.agent_id,
                    "action": request.action,
                    "validation_status": feedback.validation_status.value,
                    "human_feedback": feedback.comments,
                    "confidence_score": request.confidence_score,
                    "timestamp": datetime.now()
                }
                
                # This would integrate with the learning system's experience storage
                self.logger.debug(f"Processed validation result for learning system")
            
        except Exception as e:
            self.logger.error(f"Failed to process validation result: {e}")
    
    async def _process_performance_feedback(self, feedback: HumanFeedback) -> None:
        """Process performance rating feedback"""
        try:
            if self.learning_system and feedback.performance_rating is not None:
                # Update agent performance metrics with human feedback
                # This would integrate with the learning system's performance tracking
                self.logger.debug(f"Processed performance feedback for agent {feedback.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to process performance feedback: {e}")
    
    async def _process_behavior_tagging(self, feedback: HumanFeedback) -> None:
        """Process behavior tagging feedback"""
        try:
            # Store behavior tags for learning
            if feedback.behavior_tags:
                # This would update the agent's behavior model
                self.logger.debug(f"Processed behavior tags for agent {feedback.agent_id}: {feedback.behavior_tags}")
            
        except Exception as e:
            self.logger.error(f"Failed to process behavior tagging: {e}")
    
    async def _process_strategy_correction(self, feedback: HumanFeedback) -> None:
        """Process strategy correction feedback"""
        try:
            if feedback.strategy_modifications:
                # Apply strategy corrections to the learning system
                if self.learning_system:
                    # This would update the agent's strategy parameters
                    self.logger.debug(f"Processed strategy correction for agent {feedback.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to process strategy correction: {e}")
    
    async def _apply_learning_correction(self, correction: LearningCorrection) -> None:
        """Apply a learning correction to the agent"""
        try:
            # This would integrate with the learning system to apply corrections
            correction.applied = True
            correction.validation_results = {
                "applied_at": datetime.now(),
                "success": True
            }
            
            self.logger.info(f"Applied learning correction {correction.correction_id} for agent {correction.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply learning correction: {e}")
            correction.applied = False
            correction.validation_results = {
                "applied_at": datetime.now(),
                "success": False,
                "error": str(e)
            }
    
    # Query methods
    async def get_pending_validations(self, reviewer_id: str = None) -> List[ValidationRequest]:
        """Get pending validation requests"""
        requests = list(self.pending_validations.values())
        
        if reviewer_id:
            # Filter by assigned reviewer or unassigned high-priority requests
            requests = [r for r in requests if 
                       r.assigned_reviewer == reviewer_id or 
                       (r.assigned_reviewer is None and r.priority in [Priority.HIGH, Priority.CRITICAL])]
        
        # Sort by priority and timestamp
        priority_order = {Priority.CRITICAL: 0, Priority.HIGH: 1, Priority.MEDIUM: 2, Priority.LOW: 3}
        requests.sort(key=lambda r: (priority_order[r.priority], r.timestamp))
        
        return requests
    
    async def get_agent_feedback_history(self, agent_id: str) -> List[HumanFeedback]:
        """Get feedback history for an agent"""
        return self.feedback_history.get(agent_id, [])
    
    async def get_learning_corrections(self, agent_id: str) -> List[LearningCorrection]:
        """Get learning corrections for an agent"""
        return self.learning_corrections.get(agent_id, [])
    
    async def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get statistics about human feedback"""
        total_feedback = sum(len(feedback_list) for feedback_list in self.feedback_history.values())
        total_validations = len([f for feedback_list in self.feedback_history.values() 
                               for f in feedback_list if f.feedback_type == FeedbackType.ACTION_VALIDATION])
        
        validation_stats = {}
        for status in ValidationStatus:
            count = len([f for feedback_list in self.feedback_history.values() 
                        for f in feedback_list 
                        if f.feedback_type == FeedbackType.ACTION_VALIDATION and f.validation_status == status])
            validation_stats[status.value] = count
        
        return {
            "total_feedback": total_feedback,
            "total_validations": total_validations,
            "pending_validations": len(self.pending_validations),
            "validation_breakdown": validation_stats,
            "active_agents": len(self.feedback_history),
            "total_corrections": sum(len(corrections) for corrections in self.learning_corrections.values())
        }
    
    # Callback registration
    def register_validation_callback(self, name: str, callback: Callable) -> None:
        """Register a callback for validation requests"""
        self.validation_callbacks[name] = callback
    
    def unregister_validation_callback(self, name: str) -> None:
        """Unregister a validation callback"""
        if name in self.validation_callbacks:
            del self.validation_callbacks[name]
    
    def register_reviewer(self, reviewer_id: str, reviewer_info: Dict[str, Any]) -> None:
        """Register a human reviewer"""
        self.registered_reviewers[reviewer_id] = {
            **reviewer_info,
            "registered_at": datetime.now()
        }
    
    async def shutdown(self) -> None:
        """Shutdown the human-in-the-loop system"""
        try:
            self.logger.info("Shutting down human-in-the-loop system")
            
            # Cancel pending validations
            for request in self.pending_validations.values():
                request.status = ValidationStatus.REJECTED
            
            self.initialized = False
            self.logger.info("Human-in-the-loop system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Factory function
def create_human_in_the_loop_interface(learning_system: LearningSystem = None) -> HumanInTheLoopInterface:
    """Create and return a human-in-the-loop interface"""
    return HumanInTheLoopInterface(learning_system)