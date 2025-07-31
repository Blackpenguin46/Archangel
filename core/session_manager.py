#!/usr/bin/env python3
"""
Archangel Linux - Session Management System
Core session management infrastructure with HuggingFace integration
"""

import uuid
import time
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
from datetime import datetime, timedelta

# HuggingFace imports for model management
try:
    from huggingface_hub import HfApi, login, logout, whoami
    from huggingface_hub.utils import HfHubHTTPError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

logger = logging.getLogger(__name__)

class InteractionMode(Enum):
    """Different interaction modes in Archangel"""
    ANALYSIS = "analysis"
    CHAT = "chat"
    EXPLANATION = "explanation"
    HELP = "help"
    MENU = "menu"
    TRAINING = "training"
    MODEL_MANAGEMENT = "model_management"

class SessionStatus(Enum):
    """Session status states"""
    ACTIVE = "active"
    IDLE = "idle"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"

@dataclass
class SessionState:
    """Session state tracking"""
    session_id: str
    current_mode: InteractionMode
    user_id: str
    created_at: datetime
    last_activity: datetime
    status: SessionStatus
    preferences: Dict[str, Any]
    hf_authenticated: bool = False
    current_model: Optional[str] = None
    training_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'session_id': self.session_id,
            'current_mode': self.current_mode.value,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'status': self.status.value,
            'preferences': self.preferences,
            'hf_authenticated': self.hf_authenticated,
            'current_model': self.current_model,
            'training_context': self.training_context
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionState':
        """Create from dictionary"""
        return cls(
            session_id=data['session_id'],
            current_mode=InteractionMode(data['current_mode']),
            user_id=data['user_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_activity=datetime.fromisoformat(data['last_activity']),
            status=SessionStatus(data['status']),
            preferences=data['preferences'],
            hf_authenticated=data.get('hf_authenticated', False),
            current_model=data.get('current_model'),
            training_context=data.get('training_context')
        )

@dataclass
class AnalysisContext:
    """Analysis context for maintaining analysis history"""
    current_analysis: Optional[Dict[str, Any]]
    analysis_history: List[Dict[str, Any]]
    pending_questions: List[str]
    available_actions: List[str]
    context_metadata: Dict[str, Any]
    model_used: Optional[str] = None
    training_data_used: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisContext':
        """Create from dictionary"""
        return cls(**data)

class HuggingFaceManager:
    """Manages HuggingFace integration for models and datasets"""
    
    def __init__(self):
        self.api = HfApi() if HF_AVAILABLE else None
        self.authenticated = False
        self.current_user = None
        
    async def authenticate(self, token: str) -> bool:
        """Authenticate with HuggingFace"""
        if not HF_AVAILABLE:
            logger.warning("HuggingFace Hub not available")
            return False
            
        try:
            login(token=token)
            self.current_user = whoami()
            self.authenticated = True
            logger.info(f"Authenticated with HuggingFace as: {self.current_user['name']}")
            return True
        except Exception as e:
            logger.error(f"HuggingFace authentication failed: {e}")
            return False
    
    async def get_available_models(self, task: str = None) -> List[Dict[str, Any]]:
        """Get available models for security tasks"""
        if not self.authenticated or not self.api:
            return []
            
        try:
            # Get models suitable for security analysis
            models = list(self.api.list_models(
                filter=task,
                sort="downloads",
                direction=-1,
                limit=20
            ))
            
            return [
                {
                    'id': model.modelId,
                    'downloads': model.downloads,
                    'tags': model.tags,
                    'pipeline_tag': model.pipeline_tag,
                    'library_name': model.library_name
                }
                for model in models
            ]
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []
    
    async def get_security_datasets(self) -> List[Dict[str, Any]]:
        """Get datasets suitable for security training"""
        if not self.authenticated or not self.api:
            return []
            
        try:
            # Search for security-related datasets
            datasets = list(self.api.list_datasets(
                search="security vulnerability malware",
                sort="downloads",
                direction=-1,
                limit=10
            ))
            
            return [
                {
                    'id': dataset.id,
                    'downloads': dataset.downloads,
                    'tags': dataset.tags,
                    'description': getattr(dataset, 'description', '')
                }
                for dataset in datasets
            ]
        except Exception as e:
            logger.error(f"Failed to get datasets: {e}")
            return []
    
    async def download_model(self, model_id: str, local_dir: str = None) -> bool:
        """Download a model for local use"""
        if not self.authenticated or not self.api:
            return False
            
        try:
            from huggingface_hub import snapshot_download
            
            local_path = snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            
            logger.info(f"Downloaded model {model_id} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            return False
    
    async def download_dataset(self, dataset_id: str, local_dir: str = None) -> bool:
        """Download a dataset for training"""
        if not self.authenticated or not self.api:
            return False
            
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(dataset_id)
            
            if local_dir:
                dataset.save_to_disk(local_dir)
                logger.info(f"Downloaded dataset {dataset_id} to {local_dir}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to download dataset {dataset_id}: {e}")
            return False

class ContextStore:
    """Stores and manages analysis context and session data"""
    
    def __init__(self, storage_dir: str = "data/sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.contexts: Dict[str, AnalysisContext] = {}
        
    def store_analysis(self, session_id: str, analysis: Dict[str, Any]) -> None:
        """Store analysis results in context"""
        if session_id not in self.contexts:
            self.contexts[session_id] = AnalysisContext(
                current_analysis=None,
                analysis_history=[],
                pending_questions=[],
                available_actions=[],
                context_metadata={}
            )
        
        context = self.contexts[session_id]
        
        # Move current analysis to history
        if context.current_analysis:
            context.analysis_history.append(context.current_analysis)
        
        # Set new current analysis
        context.current_analysis = analysis
        
        # Update available actions based on analysis
        context.available_actions = self._generate_actions(analysis)
        
        # Persist to disk
        self._save_context(session_id, context)
    
    def get_analysis_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get analysis history for session"""
        if session_id in self.contexts:
            return self.contexts[session_id].analysis_history
        return []
    
    def get_current_context(self, session_id: str) -> Optional[AnalysisContext]:
        """Get current analysis context"""
        return self.contexts.get(session_id)
    
    def clear_context(self, session_id: str) -> None:
        """Clear context for session"""
        if session_id in self.contexts:
            del self.contexts[session_id]
        
        context_file = self.storage_dir / f"{session_id}_context.json"
        if context_file.exists():
            context_file.unlink()
    
    def add_pending_question(self, session_id: str, question: str) -> None:
        """Add a pending question to context"""
        if session_id in self.contexts:
            self.contexts[session_id].pending_questions.append(question)
            self._save_context(session_id, self.contexts[session_id])
    
    def get_pending_questions(self, session_id: str) -> List[str]:
        """Get pending questions for session"""
        if session_id in self.contexts:
            return self.contexts[session_id].pending_questions
        return []
    
    def clear_pending_questions(self, session_id: str) -> None:
        """Clear pending questions"""
        if session_id in self.contexts:
            self.contexts[session_id].pending_questions.clear()
            self._save_context(session_id, self.contexts[session_id])
    
    def _generate_actions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate available actions based on analysis"""
        actions = [
            "ask_followup",
            "explain_reasoning", 
            "analyze_another",
            "get_recommendations",
            "view_history"
        ]
        
        # Add specific actions based on analysis type
        if analysis.get('target_type') == 'web_application':
            actions.extend([
                "web_scan_details",
                "vulnerability_details",
                "mitigation_steps"
            ])
        elif analysis.get('target_type') == 'network':
            actions.extend([
                "port_scan_details",
                "service_enumeration",
                "network_topology"
            ])
        
        return actions
    
    def _save_context(self, session_id: str, context: AnalysisContext) -> None:
        """Save context to disk"""
        try:
            context_file = self.storage_dir / f"{session_id}_context.json"
            with open(context_file, 'w') as f:
                json.dump(context.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save context for session {session_id}: {e}")
    
    def _load_context(self, session_id: str) -> Optional[AnalysisContext]:
        """Load context from disk"""
        try:
            context_file = self.storage_dir / f"{session_id}_context.json"
            if context_file.exists():
                with open(context_file, 'r') as f:
                    data = json.load(f)
                return AnalysisContext.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load context for session {session_id}: {e}")
        return None

class SessionManager:
    """Main session manager with HuggingFace integration"""
    
    def __init__(self, storage_dir: str = "data/sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, SessionState] = {}
        self.context_store = ContextStore(storage_dir)
        self.hf_manager = HuggingFaceManager()
        
        # Load existing sessions
        self._load_sessions()
    
    def create_session(self, user_id: str = "default") -> str:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session_state = SessionState(
            session_id=session_id,
            current_mode=InteractionMode.MENU,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            status=SessionStatus.ACTIVE,
            preferences={
                'show_reasoning': True,
                'auto_continue': False,
                'preferred_models': [],
                'training_enabled': False
            }
        )
        
        self.sessions[session_id] = session_state
        self._save_session(session_state)
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID"""
        session = self.sessions.get(session_id)
        if session:
            # Update last activity
            session.last_activity = datetime.now()
            self._save_session(session)
        return session
    
    def update_session_state(self, session_id: str, **updates) -> bool:
        """Update session state"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        for key, value in updates.items():
            if hasattr(session, key):
                setattr(session, key, value)
        
        session.last_activity = datetime.now()
        self._save_session(session)
        return True
    
    def preserve_context(self, session_id: str, analysis: Dict[str, Any]) -> None:
        """Preserve analysis context"""
        self.context_store.store_analysis(session_id, analysis)
    
    def get_context(self, session_id: str) -> Optional[AnalysisContext]:
        """Get analysis context for session"""
        return self.context_store.get_current_context(session_id)
    
    def cleanup_session(self, session_id: str) -> None:
        """Clean up session"""
        if session_id in self.sessions:
            self.sessions[session_id].status = SessionStatus.TERMINATED
            self._save_session(self.sessions[session_id])
            del self.sessions[session_id]
        
        self.context_store.clear_context(session_id)
        logger.info(f"Cleaned up session: {session_id}")
    
    async def authenticate_huggingface(self, session_id: str, token: str) -> bool:
        """Authenticate session with HuggingFace"""
        success = await self.hf_manager.authenticate(token)
        
        if success and session_id in self.sessions:
            self.sessions[session_id].hf_authenticated = True
            self._save_session(self.sessions[session_id])
        
        return success
    
    async def get_available_models(self, session_id: str, task: str = None) -> List[Dict[str, Any]]:
        """Get available HuggingFace models for session"""
        if session_id not in self.sessions or not self.sessions[session_id].hf_authenticated:
            return []
        
        return await self.hf_manager.get_available_models(task)
    
    async def set_current_model(self, session_id: str, model_id: str) -> bool:
        """Set current model for session"""
        if session_id not in self.sessions:
            return False
        
        self.sessions[session_id].current_model = model_id
        self._save_session(self.sessions[session_id])
        return True
    
    def get_active_sessions(self) -> List[SessionState]:
        """Get all active sessions"""
        return [s for s in self.sessions.values() if s.status == SessionStatus.ACTIVE]
    
    def cleanup_expired_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up expired sessions"""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if session.last_activity < cutoff:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.cleanup_session(session_id)
        
        return len(expired_sessions)
    
    def _save_session(self, session: SessionState) -> None:
        """Save session to disk"""
        try:
            session_file = self.storage_dir / f"{session.session_id}.json"
            with open(session_file, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
    
    def _load_sessions(self) -> None:
        """Load sessions from disk"""
        try:
            for session_file in self.storage_dir.glob("*.json"):
                if session_file.name.endswith("_context.json"):
                    continue  # Skip context files
                
                with open(session_file, 'r') as f:
                    data = json.load(f)
                
                session = SessionState.from_dict(data)
                self.sessions[session.session_id] = session
                
                logger.info(f"Loaded session: {session.session_id}")
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")

# Global session manager instance
_session_manager = None

def get_session_manager() -> SessionManager:
    """Get global session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager