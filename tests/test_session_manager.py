#!/usr/bin/env python3
"""
Unit tests for Archangel Session Management System
"""

import pytest
import tempfile
import shutil
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.session_manager import (
    SessionManager, ContextStore, HuggingFaceManager,
    SessionState, AnalysisContext, InteractionMode, SessionStatus
)

class TestSessionManager:
    """Test SessionManager functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.session_manager = SessionManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_create_session(self):
        """Test session creation"""
        session_id = self.session_manager.create_session("test_user")
        
        assert session_id is not None
        assert len(session_id) > 0
        assert session_id in self.session_manager.sessions
        
        session = self.session_manager.get_session(session_id)
        assert session is not None
        assert session.user_id == "test_user"
        assert session.current_mode == InteractionMode.MENU
        assert session.status == SessionStatus.ACTIVE
    
    def test_session_persistence(self):
        """Test session persistence to disk"""
        session_id = self.session_manager.create_session("test_user")
        
        # Create new session manager to test loading
        new_manager = SessionManager(self.temp_dir)
        
        loaded_session = new_manager.get_session(session_id)
        assert loaded_session is not None
        assert loaded_session.user_id == "test_user"
    
    def test_update_session_state(self):
        """Test session state updates"""
        session_id = self.session_manager.create_session("test_user")
        
        success = self.session_manager.update_session_state(
            session_id,
            current_mode=InteractionMode.ANALYSIS,
            preferences={'test': 'value'}
        )
        
        assert success is True
        
        session = self.session_manager.get_session(session_id)
        assert session.current_mode == InteractionMode.ANALYSIS
        assert session.preferences['test'] == 'value'
    
    def test_preserve_context(self):
        """Test context preservation"""
        session_id = self.session_manager.create_session("test_user")
        
        analysis = {
            'target': 'example.com',
            'result': 'test analysis',
            'timestamp': datetime.now().isoformat()
        }
        
        self.session_manager.preserve_context(session_id, analysis)
        
        context = self.session_manager.get_context(session_id)
        assert context is not None
        assert context.current_analysis == analysis
    
    def test_cleanup_session(self):
        """Test session cleanup"""
        session_id = self.session_manager.create_session("test_user")
        
        # Add some context
        analysis = {'target': 'test.com', 'result': 'test'}
        self.session_manager.preserve_context(session_id, analysis)
        
        # Cleanup
        self.session_manager.cleanup_session(session_id)
        
        # Verify cleanup
        assert session_id not in self.session_manager.sessions
        context = self.session_manager.get_context(session_id)
        assert context is None
    
    def test_cleanup_expired_sessions(self):
        """Test expired session cleanup"""
        session_id = self.session_manager.create_session("test_user")
        
        # Manually set old last_activity
        session = self.session_manager.sessions[session_id]
        session.last_activity = datetime.now() - timedelta(hours=25)
        
        # Cleanup expired sessions
        cleaned = self.session_manager.cleanup_expired_sessions(max_age_hours=24)
        
        assert cleaned == 1
        assert session_id not in self.session_manager.sessions

class TestContextStore:
    """Test ContextStore functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.context_store = ContextStore(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_store_analysis(self):
        """Test analysis storage"""
        session_id = "test_session"
        analysis = {
            'target': 'example.com',
            'target_type': 'web_application',
            'result': 'test analysis'
        }
        
        self.context_store.store_analysis(session_id, analysis)
        
        context = self.context_store.get_current_context(session_id)
        assert context is not None
        assert context.current_analysis == analysis
        assert len(context.available_actions) > 0
    
    def test_analysis_history(self):
        """Test analysis history tracking"""
        session_id = "test_session"
        
        # Store first analysis
        analysis1 = {'target': 'example.com', 'result': 'first'}
        self.context_store.store_analysis(session_id, analysis1)
        
        # Store second analysis
        analysis2 = {'target': 'test.com', 'result': 'second'}
        self.context_store.store_analysis(session_id, analysis2)
        
        # Check history
        history = self.context_store.get_analysis_history(session_id)
        assert len(history) == 1
        assert history[0] == analysis1
        
        # Check current
        context = self.context_store.get_current_context(session_id)
        assert context.current_analysis == analysis2
    
    def test_pending_questions(self):
        """Test pending questions management"""
        session_id = "test_session"
        
        # Add questions
        self.context_store.add_pending_question(session_id, "What is SQL injection?")
        self.context_store.add_pending_question(session_id, "How to prevent XSS?")
        
        # Get questions
        questions = self.context_store.get_pending_questions(session_id)
        assert len(questions) == 2
        assert "What is SQL injection?" in questions
        assert "How to prevent XSS?" in questions
        
        # Clear questions
        self.context_store.clear_pending_questions(session_id)
        questions = self.context_store.get_pending_questions(session_id)
        assert len(questions) == 0
    
    def test_context_persistence(self):
        """Test context persistence to disk"""
        session_id = "test_session"
        analysis = {'target': 'example.com', 'result': 'test'}
        
        self.context_store.store_analysis(session_id, analysis)
        
        # Create new context store to test loading
        new_store = ContextStore(self.temp_dir)
        
        # Context should be loaded from disk
        context = new_store._load_context(session_id)
        assert context is not None
        assert context.current_analysis == analysis

class TestHuggingFaceManager:
    """Test HuggingFaceManager functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.hf_manager = HuggingFaceManager()
    
    @pytest.mark.asyncio
    async def test_authentication_failure(self):
        """Test authentication with invalid token"""
        success = await self.hf_manager.authenticate("invalid_token")
        assert success is False
        assert self.hf_manager.authenticated is False
    
    @pytest.mark.asyncio
    async def test_get_models_without_auth(self):
        """Test getting models without authentication"""
        models = await self.hf_manager.get_available_models()
        assert isinstance(models, list)
        assert len(models) == 0  # Should be empty without auth
    
    @pytest.mark.asyncio
    async def test_get_datasets_without_auth(self):
        """Test getting datasets without authentication"""
        datasets = await self.hf_manager.get_security_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) == 0  # Should be empty without auth

class TestSessionState:
    """Test SessionState data model"""
    
    def test_session_state_serialization(self):
        """Test SessionState to/from dict conversion"""
        now = datetime.now()
        
        state = SessionState(
            session_id="test_id",
            current_mode=InteractionMode.ANALYSIS,
            user_id="test_user",
            created_at=now,
            last_activity=now,
            status=SessionStatus.ACTIVE,
            preferences={'test': 'value'},
            hf_authenticated=True,
            current_model="test_model"
        )
        
        # Convert to dict
        state_dict = state.to_dict()
        assert state_dict['session_id'] == "test_id"
        assert state_dict['current_mode'] == "analysis"
        assert state_dict['hf_authenticated'] is True
        
        # Convert back from dict
        restored_state = SessionState.from_dict(state_dict)
        assert restored_state.session_id == state.session_id
        assert restored_state.current_mode == state.current_mode
        assert restored_state.hf_authenticated == state.hf_authenticated

class TestAnalysisContext:
    """Test AnalysisContext data model"""
    
    def test_analysis_context_serialization(self):
        """Test AnalysisContext to/from dict conversion"""
        context = AnalysisContext(
            current_analysis={'target': 'test.com'},
            analysis_history=[{'target': 'old.com'}],
            pending_questions=['What is XSS?'],
            available_actions=['explain', 'analyze'],
            context_metadata={'session': 'test'},
            model_used='test_model',
            training_data_used=['dataset1', 'dataset2']
        )
        
        # Convert to dict
        context_dict = context.to_dict()
        assert context_dict['current_analysis']['target'] == 'test.com'
        assert len(context_dict['analysis_history']) == 1
        assert context_dict['model_used'] == 'test_model'
        
        # Convert back from dict
        restored_context = AnalysisContext.from_dict(context_dict)
        assert restored_context.current_analysis == context.current_analysis
        assert restored_context.model_used == context.model_used

# Integration tests
class TestSessionIntegration:
    """Integration tests for session management"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.session_manager = SessionManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_session_workflow(self):
        """Test complete session workflow"""
        # Create session
        session_id = self.session_manager.create_session("test_user")
        
        # Update mode to analysis
        self.session_manager.update_session_state(
            session_id,
            current_mode=InteractionMode.ANALYSIS
        )
        
        # Store analysis
        analysis = {
            'target': 'example.com',
            'target_type': 'web_application',
            'vulnerabilities': ['xss', 'sql_injection'],
            'recommendations': ['input_validation', 'output_encoding']
        }
        self.session_manager.preserve_context(session_id, analysis)
        
        # Add follow-up questions
        context_store = self.session_manager.context_store
        context_store.add_pending_question(session_id, "How to fix XSS?")
        context_store.add_pending_question(session_id, "What about SQL injection?")
        
        # Verify complete state
        session = self.session_manager.get_session(session_id)
        assert session.current_mode == InteractionMode.ANALYSIS
        
        context = self.session_manager.get_context(session_id)
        assert context.current_analysis == analysis
        assert len(context.available_actions) > 0
        
        questions = context_store.get_pending_questions(session_id)
        assert len(questions) == 2
        
        # Test persistence by creating new manager
        new_manager = SessionManager(self.temp_dir)
        
        loaded_session = new_manager.get_session(session_id)
        assert loaded_session is not None
        assert loaded_session.current_mode == InteractionMode.ANALYSIS
        
        loaded_context = new_manager.get_context(session_id)
        assert loaded_context is not None
        assert loaded_context.current_analysis == analysis

if __name__ == "__main__":
    pytest.main([__file__, "-v"])