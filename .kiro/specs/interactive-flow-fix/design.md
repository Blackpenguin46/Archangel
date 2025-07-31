# Design Document

## Overview

The interactive flow fix addresses session management and user experience issues in Archangel's interactive modes. The core problem is that the current implementation doesn't properly maintain session state and context after target analysis, leading to workflow interruptions and poor user experience.

The solution involves implementing a robust session management system with proper state handling, context preservation, and consistent user interface patterns across all interactive modes.

## Architecture

### Session Management Layer
- **SessionManager**: Central coordinator for all interactive sessions
- **ContextStore**: Maintains analysis history and current session state
- **PromptController**: Manages consistent prompt behavior and transitions
- **InteractionHandler**: Processes user input and maintains workflow continuity

### State Management
- **SessionState**: Tracks current mode, analysis context, and user preferences
- **AnalysisContext**: Stores completed analyses and their results
- **UserSession**: Maintains user interaction history and preferences

### Flow Control
- **WorkflowEngine**: Manages transitions between different interaction modes
- **ContinuationManager**: Handles post-analysis options and follow-up actions
- **ErrorHandler**: Provides graceful error recovery and session preservation

## Components and Interfaces

### SessionManager Interface
```python
class SessionManager:
    def create_session(self, user_id: str) -> Session
    def get_session(self, session_id: str) -> Session
    def update_session_state(self, session_id: str, state: SessionState)
    def preserve_context(self, session_id: str, context: AnalysisContext)
    def cleanup_session(self, session_id: str)
```

### ContextStore Interface
```python
class ContextStore:
    def store_analysis(self, session_id: str, analysis: SecurityAnalysis)
    def get_analysis_history(self, session_id: str) -> List[SecurityAnalysis]
    def get_current_context(self, session_id: str) -> AnalysisContext
    def clear_context(self, session_id: str)
```

### PromptController Interface
```python
class PromptController:
    def get_prompt(self, mode: InteractionMode, context: AnalysisContext) -> str
    def show_continuation_options(self, analysis: SecurityAnalysis) -> List[str]
    def handle_mode_transition(self, from_mode: str, to_mode: str)
    def format_error_prompt(self, error: Exception) -> str
```

### InteractionHandler Interface
```python
class InteractionHandler:
    def process_input(self, session_id: str, user_input: str) -> InteractionResult
    def handle_analysis_completion(self, session_id: str, analysis: SecurityAnalysis)
    def handle_follow_up_question(self, session_id: str, question: str) -> str
    def handle_session_interruption(self, session_id: str, interruption: Exception)
```

## Data Models

### SessionState
```python
@dataclass
class SessionState:
    session_id: str
    current_mode: InteractionMode
    user_id: str
    created_at: datetime
    last_activity: datetime
    is_active: bool
    preferences: Dict[str, Any]
```

### AnalysisContext
```python
@dataclass
class AnalysisContext:
    current_analysis: Optional[SecurityAnalysis]
    analysis_history: List[SecurityAnalysis]
    pending_questions: List[str]
    available_actions: List[str]
    context_metadata: Dict[str, Any]
```

### InteractionResult
```python
@dataclass
class InteractionResult:
    response: str
    new_state: SessionState
    continuation_options: List[str]
    requires_follow_up: bool
    error: Optional[Exception]
```

### InteractionMode
```python
class InteractionMode(Enum):
    ANALYSIS = "analysis"
    CHAT = "chat"
    EXPLANATION = "explanation"
    HELP = "help"
    MENU = "menu"
```

## Error Handling

### Session Recovery
- Implement automatic session recovery for unexpected interruptions
- Maintain session checkpoints at key interaction points
- Provide graceful degradation when context is partially lost

### Error Classification
- **Recoverable Errors**: Network timeouts, temporary API failures
- **User Errors**: Invalid commands, malformed input
- **System Errors**: Internal failures, resource exhaustion

### Error Response Strategy
- Always return to a stable interactive state
- Preserve as much context as possible
- Provide clear error messages and recovery options
- Log errors for debugging while maintaining user experience

## Testing Strategy

### Unit Tests
- Test each component interface independently
- Mock external dependencies (AI models, network calls)
- Verify state transitions and context preservation
- Test error handling and recovery scenarios

### Integration Tests
- Test complete user workflows end-to-end
- Verify session persistence across mode transitions
- Test concurrent session handling
- Validate context preservation during interruptions

### User Experience Tests
- Test common user interaction patterns
- Verify prompt consistency across modes
- Test session recovery scenarios
- Validate continuation option functionality

### Performance Tests
- Test session management under load
- Verify context storage efficiency
- Test memory usage with long-running sessions
- Validate cleanup and garbage collection

## Implementation Phases

### Phase 1: Core Session Management
- Implement SessionManager and basic state tracking
- Create ContextStore for analysis preservation
- Establish basic prompt consistency

### Phase 2: Flow Control
- Implement WorkflowEngine for mode transitions
- Add ContinuationManager for post-analysis options
- Create robust error handling and recovery

### Phase 3: User Experience Enhancement
- Implement consistent prompting across all modes
- Add session history and context awareness
- Create comprehensive help and guidance system

### Phase 4: Advanced Features
- Add session persistence across application restarts
- Implement user preferences and customization
- Add advanced context search and retrieval