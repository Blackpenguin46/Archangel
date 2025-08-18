#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Critical Path Optimization and MVP Identification
System for analyzing task dependencies, identifying critical paths, and managing MVP delivery
"""

import logging
import json
import yaml
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels"""
    MUST = "MUST"         # Critical path tasks for MVP
    SHOULD = "SHOULD"     # Important features for full release
    COULD = "COULD"       # Advanced features for future versions
    WONT = "WONT"         # Features deferred to later versions

class TaskStatus(Enum):
    """Task completion status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    ON_HOLD = "on_hold"

class DeliveryPhase(Enum):
    """Delivery phases for incremental rollout"""
    FOUNDATION = "foundation"      # Core architecture and basic functionality
    ALPHA = "alpha"               # MVP with essential features
    BETA = "beta"                 # Feature-complete beta version
    PRODUCTION = "production"     # Production-ready release

@dataclass
class TaskDefinition:
    """Definition of a project task"""
    task_id: str
    name: str
    description: str
    
    # Dependencies and relationships
    dependencies: List[str] = field(default_factory=list)
    blocking_tasks: List[str] = field(default_factory=list)
    
    # Prioritization
    priority: TaskPriority = TaskPriority.SHOULD
    delivery_phase: DeliveryPhase = DeliveryPhase.BETA
    
    # Resource estimation
    estimated_effort_hours: float = 8.0
    estimated_duration_days: float = 1.0
    complexity_score: float = 1.0  # 1-5 scale
    
    # Status tracking
    status: TaskStatus = TaskStatus.NOT_STARTED
    progress_percentage: float = 0.0
    
    # Metadata
    category: str = ""
    tags: Set[str] = field(default_factory=set)
    assignee: Optional[str] = None
    sprint: Optional[int] = None
    
    # Quality gates
    acceptance_criteria: List[str] = field(default_factory=list)
    testing_requirements: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Risk assessment
    risk_level: float = 1.0  # 1-5 scale
    risk_factors: List[str] = field(default_factory=list)

@dataclass
class CriticalPath:
    """Critical path analysis result"""
    path_id: str
    tasks: List[str]
    total_duration: float
    total_effort: float
    start_date: datetime
    end_date: datetime
    
    # Analysis metrics
    slack_time: float = 0.0
    criticality_score: float = 1.0
    risk_score: float = 1.0
    
    # Path characteristics
    is_critical: bool = False
    bottleneck_tasks: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived metrics"""
        self.is_critical = self.slack_time <= 0.0

@dataclass
class MVPDefinition:
    """Minimum Viable Product definition"""
    mvp_id: str
    name: str
    description: str
    
    # Core features
    required_tasks: List[str] = field(default_factory=list)
    core_features: List[str] = field(default_factory=list)
    
    # Success criteria
    acceptance_criteria: List[str] = field(default_factory=list)
    quality_gates: List[str] = field(default_factory=list)
    
    # Delivery targets
    target_delivery_date: Optional[datetime] = None
    estimated_effort_hours: float = 0.0
    estimated_duration_days: float = 0.0
    
    # Risk assessment
    risk_factors: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)

class CriticalPathAnalyzer:
    """
    Critical path analysis and optimization system.
    
    Features:
    - Dependency graph analysis and visualization
    - Critical path identification using CPM (Critical Path Method)
    - Resource optimization and bottleneck identification
    - Risk analysis and mitigation planning
    - MVP identification and phasing
    """
    
    def __init__(self):
        self.tasks: Dict[str, TaskDefinition] = {}
        self.dependency_graph = nx.DiGraph()
        self.critical_paths: List[CriticalPath] = []
        self.mvp_definition: Optional[MVPDefinition] = None
        self.logger = logging.getLogger(__name__)
    
    def load_tasks_from_yaml(self, yaml_path: Path) -> None:
        """Load task definitions from YAML file"""
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            tasks_data = data.get('tasks', [])
            
            for task_data in tasks_data:
                task = TaskDefinition(
                    task_id=task_data['task_id'],
                    name=task_data['name'],
                    description=task_data['description'],
                    dependencies=task_data.get('dependencies', []),
                    priority=TaskPriority(task_data.get('priority', 'SHOULD')),
                    delivery_phase=DeliveryPhase(task_data.get('delivery_phase', 'BETA')),
                    estimated_effort_hours=task_data.get('estimated_effort_hours', 8.0),
                    estimated_duration_days=task_data.get('estimated_duration_days', 1.0),
                    complexity_score=task_data.get('complexity_score', 1.0),
                    status=TaskStatus(task_data.get('status', 'NOT_STARTED')),
                    category=task_data.get('category', ''),
                    tags=set(task_data.get('tags', [])),
                    acceptance_criteria=task_data.get('acceptance_criteria', []),
                    testing_requirements=task_data.get('testing_requirements', []),
                    risk_level=task_data.get('risk_level', 1.0),
                    risk_factors=task_data.get('risk_factors', [])
                )
                
                self.tasks[task.task_id] = task
            
            self._build_dependency_graph()
            self.logger.info(f"Loaded {len(self.tasks)} tasks from {yaml_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load tasks from YAML: {e}")
            raise
    
    def add_task(self, task: TaskDefinition) -> None:
        """Add a task to the system"""
        self.tasks[task.task_id] = task
        self._update_dependency_graph()
    
    def _build_dependency_graph(self) -> None:
        """Build the dependency graph from tasks"""
        self.dependency_graph.clear()
        
        # Add all tasks as nodes
        for task_id, task in self.tasks.items():
            self.dependency_graph.add_node(
                task_id,
                task=task,
                duration=task.estimated_duration_days,
                effort=task.estimated_effort_hours,
                priority=task.priority.value,
                status=task.status.value
            )
        
        # Add dependency edges
        for task_id, task in self.tasks.items():
            for dependency in task.dependencies:
                if dependency in self.tasks:
                    self.dependency_graph.add_edge(dependency, task_id)
                else:
                    self.logger.warning(f"Task {task_id} depends on unknown task {dependency}")
        
        # Update blocking relationships
        self._update_blocking_relationships()
    
    def _update_dependency_graph(self) -> None:
        """Update the dependency graph after task changes"""
        self._build_dependency_graph()
    
    def _update_blocking_relationships(self) -> None:
        """Update blocking task relationships"""
        for task_id in self.tasks:
            blocking_tasks = list(self.dependency_graph.successors(task_id))
            self.tasks[task_id].blocking_tasks = blocking_tasks
    
    def analyze_critical_paths(self) -> List[CriticalPath]:
        """Analyze critical paths using Critical Path Method (CPM)"""
        try:
            if not self.dependency_graph.nodes:
                self.logger.warning("No tasks loaded for critical path analysis")
                return []
            
            # Check for cycles
            if not nx.is_directed_acyclic_graph(self.dependency_graph):
                cycles = list(nx.simple_cycles(self.dependency_graph))
                self.logger.error(f"Dependency cycles detected: {cycles}")
                raise ValueError(f"Cannot analyze critical path with dependency cycles: {cycles}")
            
            # Find all paths from start to end nodes
            start_nodes = [n for n in self.dependency_graph.nodes() 
                          if self.dependency_graph.in_degree(n) == 0]
            end_nodes = [n for n in self.dependency_graph.nodes() 
                        if self.dependency_graph.out_degree(n) == 0]
            
            if not start_nodes or not end_nodes:
                self.logger.warning("No clear start or end nodes found")
                return []
            
            all_paths = []
            for start in start_nodes:
                for end in end_nodes:
                    try:
                        paths = list(nx.all_simple_paths(self.dependency_graph, start, end))
                        all_paths.extend(paths)
                    except nx.NetworkXNoPath:
                        continue
            
            # Analyze each path
            critical_paths = []
            max_duration = 0.0
            
            for i, path in enumerate(all_paths):
                total_duration = sum(self.dependency_graph.nodes[task]['duration'] for task in path)
                total_effort = sum(self.dependency_graph.nodes[task]['effort'] for task in path)
                
                start_date = datetime.now()
                end_date = start_date + timedelta(days=total_duration)
                
                # Calculate criticality score based on duration and priority
                priority_weight = sum(
                    3.0 if self.tasks[task].priority == TaskPriority.MUST else
                    2.0 if self.tasks[task].priority == TaskPriority.SHOULD else
                    1.0 for task in path
                )
                criticality_score = (total_duration * priority_weight) / len(path)
                
                # Calculate risk score
                risk_score = sum(self.tasks[task].risk_level for task in path) / len(path)
                
                critical_path = CriticalPath(
                    path_id=f"path_{i+1}",
                    tasks=path,
                    total_duration=total_duration,
                    total_effort=total_effort,
                    start_date=start_date,
                    end_date=end_date,
                    criticality_score=criticality_score,
                    risk_score=risk_score
                )
                
                max_duration = max(max_duration, total_duration)
                critical_paths.append(critical_path)
            
            # Calculate slack time and identify critical paths
            for path in critical_paths:
                path.slack_time = max_duration - path.total_duration
                path.is_critical = path.slack_time <= 0.5  # Allow small tolerance
            
            # Sort by criticality
            critical_paths.sort(key=lambda p: (p.is_critical, p.criticality_score), reverse=True)
            
            self.critical_paths = critical_paths
            self.logger.info(f"Identified {len(critical_paths)} paths, {sum(p.is_critical for p in critical_paths)} critical")
            
            return critical_paths
            
        except Exception as e:
            self.logger.error(f"Critical path analysis failed: {e}")
            raise
    
    def identify_bottlenecks(self) -> Dict[str, Any]:
        """Identify bottleneck tasks and resources"""
        bottlenecks = {
            'task_bottlenecks': [],
            'resource_bottlenecks': [],
            'dependency_bottlenecks': [],
            'risk_bottlenecks': []
        }
        
        try:
            # Task bottlenecks (high duration, high dependency count)
            for task_id, task in self.tasks.items():
                in_degree = self.dependency_graph.in_degree(task_id)
                out_degree = self.dependency_graph.out_degree(task_id)
                
                # High-impact tasks with many dependencies
                if in_degree > 3 or out_degree > 3:
                    bottlenecks['dependency_bottlenecks'].append({
                        'task_id': task_id,
                        'task_name': task.name,
                        'in_degree': in_degree,
                        'out_degree': out_degree,
                        'impact_score': in_degree + out_degree
                    })
                
                # Long-duration tasks
                if task.estimated_duration_days > 5.0:
                    bottlenecks['task_bottlenecks'].append({
                        'task_id': task_id,
                        'task_name': task.name,
                        'duration': task.estimated_duration_days,
                        'effort': task.estimated_effort_hours
                    })
                
                # High-risk tasks
                if task.risk_level > 3.0:
                    bottlenecks['risk_bottlenecks'].append({
                        'task_id': task_id,
                        'task_name': task.name,
                        'risk_level': task.risk_level,
                        'risk_factors': task.risk_factors
                    })
            
            # Sort bottlenecks by impact
            bottlenecks['dependency_bottlenecks'].sort(
                key=lambda x: x['impact_score'], reverse=True
            )
            bottlenecks['task_bottlenecks'].sort(
                key=lambda x: x['duration'], reverse=True
            )
            bottlenecks['risk_bottlenecks'].sort(
                key=lambda x: x['risk_level'], reverse=True
            )
            
            self.logger.info(f"Identified {sum(len(v) for v in bottlenecks.values())} bottlenecks")
            return bottlenecks
            
        except Exception as e:
            self.logger.error(f"Bottleneck identification failed: {e}")
            return bottlenecks
    
    def define_mvp(self, 
                  core_features: List[str],
                  quality_gates: List[str] = None,
                  target_date: Optional[datetime] = None) -> MVPDefinition:
        """Define the Minimum Viable Product"""
        try:
            # Identify MUST tasks for MVP
            must_tasks = [
                task_id for task_id, task in self.tasks.items()
                if task.priority == TaskPriority.MUST or 
                   task.delivery_phase == DeliveryPhase.FOUNDATION
            ]
            
            # Add dependencies of MUST tasks
            mvp_tasks = set(must_tasks)
            for task_id in must_tasks:
                predecessors = list(nx.ancestors(self.dependency_graph, task_id))
                mvp_tasks.update(predecessors)
            
            # Calculate MVP metrics
            total_effort = sum(
                self.tasks[task_id].estimated_effort_hours 
                for task_id in mvp_tasks
            )
            
            total_duration = 0.0
            if mvp_tasks:
                # Create subgraph for MVP tasks
                mvp_graph = self.dependency_graph.subgraph(mvp_tasks)
                if mvp_graph.nodes:
                    # Find longest path in MVP
                    try:
                        longest_path = nx.dag_longest_path(mvp_graph, weight='duration')
                        total_duration = nx.dag_longest_path_length(mvp_graph, weight='duration')
                    except:
                        # Fallback calculation
                        total_duration = sum(
                            self.tasks[task_id].estimated_duration_days 
                            for task_id in mvp_tasks
                        ) / len(mvp_tasks)  # Conservative estimate
            
            # Default quality gates
            if quality_gates is None:
                quality_gates = [
                    "All MUST priority tasks completed",
                    "Core functionality working end-to-end",
                    "Basic security validation passed",
                    "Essential integration tests passing",
                    "MVP acceptance criteria met"
                ]
            
            mvp = MVPDefinition(
                mvp_id="archangel_mvp_v1",
                name="Archangel MVP v1.0",
                description="Minimum viable product with core autonomous AI security features",
                required_tasks=list(mvp_tasks),
                core_features=core_features,
                quality_gates=quality_gates,
                target_delivery_date=target_date,
                estimated_effort_hours=total_effort,
                estimated_duration_days=total_duration,
                acceptance_criteria=[
                    "Basic multi-agent coordination working",
                    "Simple Red vs Blue team scenarios executable", 
                    "Memory and knowledge systems functional",
                    "Basic monitoring and logging operational",
                    "Core security controls implemented"
                ]
            )
            
            self.mvp_definition = mvp
            self.logger.info(f"Defined MVP with {len(mvp_tasks)} tasks, {total_effort:.1f}h effort, {total_duration:.1f}d duration")
            
            return mvp
            
        except Exception as e:
            self.logger.error(f"MVP definition failed: {e}")
            raise
    
    def optimize_delivery_sequence(self) -> Dict[DeliveryPhase, List[str]]:
        """Optimize task sequence for incremental delivery"""
        try:
            delivery_plan = {phase: [] for phase in DeliveryPhase}
            
            # Group tasks by delivery phase
            for task_id, task in self.tasks.items():
                delivery_plan[task.delivery_phase].append(task_id)
            
            # Optimize sequence within each phase using topological sort
            for phase, task_list in delivery_plan.items():
                if task_list:
                    # Create subgraph for phase tasks
                    phase_graph = self.dependency_graph.subgraph(task_list)
                    
                    if phase_graph.nodes:
                        try:
                            # Topologically sort tasks in phase
                            sorted_tasks = list(nx.topological_sort(phase_graph))
                            delivery_plan[phase] = sorted_tasks
                        except nx.NetworkXError:
                            # Handle cycles within phase
                            self.logger.warning(f"Dependency issues in {phase.value} phase")
            
            # Ensure dependencies across phases are respected
            optimized_plan = self._validate_cross_phase_dependencies(delivery_plan)
            
            self.logger.info("Optimized delivery sequence across phases")
            return optimized_plan
            
        except Exception as e:
            self.logger.error(f"Delivery sequence optimization failed: {e}")
            raise
    
    def _validate_cross_phase_dependencies(self, 
                                         delivery_plan: Dict[DeliveryPhase, List[str]]
                                         ) -> Dict[DeliveryPhase, List[str]]:
        """Validate and fix cross-phase dependencies"""
        phase_order = [DeliveryPhase.FOUNDATION, DeliveryPhase.ALPHA, 
                      DeliveryPhase.BETA, DeliveryPhase.PRODUCTION]
        
        validated_plan = {phase: [] for phase in DeliveryPhase}
        
        for phase in phase_order:
            for task_id in delivery_plan[phase]:
                task = self.tasks[task_id]
                
                # Check if all dependencies are in earlier phases
                dependencies_satisfied = True
                for dep_id in task.dependencies:
                    if dep_id in self.tasks:
                        dep_task = self.tasks[dep_id]
                        dep_phase_index = phase_order.index(dep_task.delivery_phase)
                        current_phase_index = phase_order.index(phase)
                        
                        if dep_phase_index > current_phase_index:
                            # Dependency is in later phase - move task or dependency
                            dependencies_satisfied = False
                            break
                
                if dependencies_satisfied:
                    validated_plan[phase].append(task_id)
                else:
                    # Move to later phase where dependencies are satisfied
                    for later_phase in phase_order[phase_order.index(phase):]:
                        deps_satisfied = all(
                            self.tasks[dep_id].delivery_phase in 
                            phase_order[:phase_order.index(later_phase)+1]
                            for dep_id in task.dependencies
                            if dep_id in self.tasks
                        )
                        if deps_satisfied:
                            validated_plan[later_phase].append(task_id)
                            break
        
        return validated_plan
    
    def generate_project_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive project dashboard data"""
        try:
            dashboard = {
                'project_overview': {
                    'total_tasks': len(self.tasks),
                    'completed_tasks': sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED),
                    'in_progress_tasks': sum(1 for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS),
                    'blocked_tasks': sum(1 for t in self.tasks.values() if t.status == TaskStatus.BLOCKED),
                    'total_estimated_effort': sum(t.estimated_effort_hours for t in self.tasks.values()),
                    'total_estimated_duration': sum(t.estimated_duration_days for t in self.tasks.values())
                },
                'priority_breakdown': {
                    priority.value: sum(1 for t in self.tasks.values() if t.priority == priority)
                    for priority in TaskPriority
                },
                'phase_breakdown': {
                    phase.value: sum(1 for t in self.tasks.values() if t.delivery_phase == phase)
                    for phase in DeliveryPhase
                },
                'critical_paths': [
                    {
                        'path_id': cp.path_id,
                        'task_count': len(cp.tasks),
                        'total_duration': cp.total_duration,
                        'is_critical': cp.is_critical,
                        'criticality_score': cp.criticality_score,
                        'risk_score': cp.risk_score
                    }
                    for cp in self.critical_paths
                ],
                'bottlenecks': self.identify_bottlenecks(),
                'mvp_status': {
                    'defined': self.mvp_definition is not None,
                    'task_count': len(self.mvp_definition.required_tasks) if self.mvp_definition else 0,
                    'estimated_effort': self.mvp_definition.estimated_effort_hours if self.mvp_definition else 0,
                    'estimated_duration': self.mvp_definition.estimated_duration_days if self.mvp_definition else 0
                }
            }
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Dashboard generation failed: {e}")
            return {}
    
    def export_analysis(self, output_path: Path) -> None:
        """Export complete analysis to files"""
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Export tasks
            tasks_data = {
                'tasks': [asdict(task) for task in self.tasks.values()]
            }
            with open(output_path / 'tasks.json', 'w') as f:
                json.dump(tasks_data, f, indent=2, default=str)
            
            # Export critical paths
            if self.critical_paths:
                paths_data = {
                    'critical_paths': [asdict(path) for path in self.critical_paths]
                }
                with open(output_path / 'critical_paths.json', 'w') as f:
                    json.dump(paths_data, f, indent=2, default=str)
            
            # Export MVP definition
            if self.mvp_definition:
                mvp_data = asdict(self.mvp_definition)
                with open(output_path / 'mvp_definition.json', 'w') as f:
                    json.dump(mvp_data, f, indent=2, default=str)
            
            # Export dashboard
            dashboard = self.generate_project_dashboard()
            with open(output_path / 'project_dashboard.json', 'w') as f:
                json.dump(dashboard, f, indent=2, default=str)
            
            self.logger.info(f"Analysis exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            raise
    
    def visualize_dependency_graph(self, output_path: Path) -> None:
        """Create dependency graph visualization"""
        try:
            plt.figure(figsize=(20, 15))
            
            # Create layout
            pos = nx.spring_layout(self.dependency_graph, k=3, iterations=50)
            
            # Color nodes by priority
            node_colors = []
            for node in self.dependency_graph.nodes():
                task = self.tasks[node]
                if task.priority == TaskPriority.MUST:
                    node_colors.append('red')
                elif task.priority == TaskPriority.SHOULD:
                    node_colors.append('orange')
                elif task.priority == TaskPriority.COULD:
                    node_colors.append('yellow')
                else:
                    node_colors.append('lightgray')
            
            # Draw graph
            nx.draw(self.dependency_graph, pos, 
                   node_color=node_colors,
                   node_size=1000,
                   font_size=8,
                   font_weight='bold',
                   arrows=True,
                   arrowsize=20,
                   edge_color='gray',
                   with_labels=True)
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                          markersize=10, label='MUST (Critical)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                          markersize=10, label='SHOULD (Important)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                          markersize=10, label='COULD (Nice-to-have)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
                          markersize=10, label='WONT (Deferred)')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.title("Archangel Project Dependency Graph", size=16, weight='bold')
            plt.tight_layout()
            plt.savefig(output_path / 'dependency_graph.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Dependency graph saved to {output_path}/dependency_graph.png")
            
        except Exception as e:
            self.logger.error(f"Graph visualization failed: {e}")


def create_archangel_task_definitions() -> List[TaskDefinition]:
    """Create task definitions for the Archangel project based on tasks.md"""
    return [
        # Foundation Layer (MUST complete first)
        TaskDefinition(
            task_id="task_01",
            name="Set up foundational multi-agent coordination framework",
            description="Implement LangGraph coordinator, base agent architecture, communication bus",
            dependencies=[],
            priority=TaskPriority.MUST,
            delivery_phase=DeliveryPhase.FOUNDATION,
            estimated_effort_hours=32.0,
            estimated_duration_days=4.0,
            complexity_score=4.0,
            status=TaskStatus.COMPLETED,
            category="Infrastructure",
            tags={"foundation", "multi-agent", "coordination"},
            acceptance_criteria=[
                "LangGraph coordinator operational",
                "Base agent architecture implemented",
                "Secure communication bus working",
                "Unit tests passing"
            ]
        ),
        
        TaskDefinition(
            task_id="task_02",
            name="Implement vector memory and knowledge base systems",
            description="Set up ChromaDB/Weaviate, memory clustering, MITRE ATT&CK integration",
            dependencies=[],
            priority=TaskPriority.MUST,
            delivery_phase=DeliveryPhase.FOUNDATION,
            estimated_effort_hours=40.0,
            estimated_duration_days=5.0,
            complexity_score=4.0,
            status=TaskStatus.COMPLETED,
            category="Memory",
            tags={"memory", "knowledge", "database"},
            acceptance_criteria=[
                "Vector database operational",
                "Memory clustering working",
                "MITRE ATT&CK integration complete",
                "Knowledge base tests passing"
            ]
        ),
        
        TaskDefinition(
            task_id="task_49",
            name="Refactor system into modular layered architecture",
            description="Create clear separation between Data, Model, Logic, and Interface layers",
            dependencies=[],
            priority=TaskPriority.MUST,
            delivery_phase=DeliveryPhase.FOUNDATION,
            estimated_effort_hours=24.0,
            estimated_duration_days=3.0,
            complexity_score=3.0,
            status=TaskStatus.COMPLETED,
            category="Architecture",
            tags={"refactoring", "architecture", "modularity"},
            acceptance_criteria=[
                "Clear layer separation implemented",
                "Interface contracts defined",
                "Layer tests passing"
            ]
        ),
        
        # Agent Development Layer
        TaskDefinition(
            task_id="task_03",
            name="Create Red Team autonomous agent implementations",
            description="Implement ReconAgent, ExploitAgent, PersistenceAgent, ExfiltrationAgent",
            dependencies=["task_01", "task_02", "task_49"],
            priority=TaskPriority.MUST,
            delivery_phase=DeliveryPhase.ALPHA,
            estimated_effort_hours=48.0,
            estimated_duration_days=6.0,
            complexity_score=4.0,
            status=TaskStatus.COMPLETED,
            category="Agents",
            tags={"red-team", "agents", "autonomous"},
            acceptance_criteria=[
                "All Red Team agents implemented",
                "Decision-making logic working",
                "Intelligence sharing functional",
                "Integration tests passing"
            ]
        ),
        
        TaskDefinition(
            task_id="task_04",
            name="Develop Blue Team autonomous agent implementations",
            description="Implement SOCAnalyst, FirewallConfigurator, SIEMIntegrator, ComplianceAuditor",
            dependencies=["task_01", "task_02", "task_49"],
            priority=TaskPriority.MUST,
            delivery_phase=DeliveryPhase.ALPHA,
            estimated_effort_hours=48.0,
            estimated_duration_days=6.0,
            complexity_score=4.0,
            status=TaskStatus.COMPLETED,
            category="Agents",
            tags={"blue-team", "agents", "autonomous"},
            acceptance_criteria=[
                "All Blue Team agents implemented",
                "Defensive logic working",
                "Response coordination functional",
                "Integration tests passing"
            ]
        ),
        
        TaskDefinition(
            task_id="task_41",
            name="Implement end-to-end autonomous simulation flow",
            description="Complete simulation pipeline from initialization to mission completion",
            dependencies=["task_03", "task_04"],
            priority=TaskPriority.MUST,
            delivery_phase=DeliveryPhase.ALPHA,
            estimated_effort_hours=40.0,
            estimated_duration_days=5.0,
            complexity_score=5.0,
            status=TaskStatus.IN_PROGRESS,
            category="Simulation",
            tags={"end-to-end", "simulation", "pipeline"},
            acceptance_criteria=[
                "Complete simulation pipeline working",
                "Agent decision-making flow operational",
                "Team coordination workflow functional",
                "Mission lifecycle management complete"
            ]
        ),
        
        TaskDefinition(
            task_id="task_42",
            name="Create structured Red and Blue team knowledge libraries",
            description="Build comprehensive tactic libraries with MITRE ATT&CK mapping",
            dependencies=["task_02"],
            priority=TaskPriority.MUST,
            delivery_phase=DeliveryPhase.ALPHA,
            estimated_effort_hours=32.0,
            estimated_duration_days=4.0,
            complexity_score=3.0,
            status=TaskStatus.COMPLETED,
            category="Knowledge",
            tags={"knowledge", "tactics", "mitre"},
            acceptance_criteria=[
                "Red Team tactic library complete",
                "Blue Team response library complete",
                "MITRE ATT&CK mapping accurate",
                "Knowledge library tests passing"
            ]
        ),
        
        # Environment Layer
        TaskDefinition(
            task_id="task_05",
            name="Build comprehensive mock enterprise environment infrastructure",
            description="Deploy containerized frontend, backend services, network segmentation",
            dependencies=["task_01"],
            priority=TaskPriority.MUST,
            delivery_phase=DeliveryPhase.ALPHA,
            estimated_effort_hours=40.0,
            estimated_duration_days=5.0,
            complexity_score=4.0,
            status=TaskStatus.COMPLETED,
            category="Environment",
            tags={"infrastructure", "containerization", "network"},
            acceptance_criteria=[
                "Containerized environment deployed",
                "Network segmentation implemented",
                "Logging infrastructure operational",
                "Infrastructure tests passing"
            ]
        ),
        
        TaskDefinition(
            task_id="task_06",
            name="Implement deception technologies and honeypot systems",
            description="Deploy multi-tier honeypots, honeytoken distribution, decoy services",
            dependencies=["task_05"],
            priority=TaskPriority.SHOULD,
            delivery_phase=DeliveryPhase.BETA,
            estimated_effort_hours=32.0,
            estimated_duration_days=4.0,
            complexity_score=3.0,
            status=TaskStatus.COMPLETED,
            category="Deception",
            tags={"honeypots", "deception", "monitoring"},
            acceptance_criteria=[
                "Multi-tier honeypots deployed",
                "Honeytoken system operational",
                "Deception effectiveness measured",
                "Monitoring and alerting working"
            ]
        ),
        
        TaskDefinition(
            task_id="task_07",
            name="Create synthetic user simulation and background activity",
            description="Implement autonomous synthetic users with realistic behavior patterns",
            dependencies=["task_05"],
            priority=TaskPriority.SHOULD,
            delivery_phase=DeliveryPhase.BETA,
            estimated_effort_hours=24.0,
            estimated_duration_days=3.0,
            complexity_score=3.0,
            status=TaskStatus.COMPLETED,
            category="Simulation",
            tags={"synthetic-users", "background-activity", "realism"},
            acceptance_criteria=[
                "Synthetic user agents working",
                "Realistic behavior patterns implemented",
                "Email and file activity simulated",
                "Web browsing simulation operational"
            ]
        ),
        
        # Game Loop and Evaluation
        TaskDefinition(
            task_id="task_10",
            name="Create phase-based game loop and scenario management",
            description="Implement FSM for phases, scenario configuration, objective tracking",
            dependencies=["task_03", "task_04", "task_05"],
            priority=TaskPriority.MUST,
            delivery_phase=DeliveryPhase.ALPHA,
            estimated_effort_hours=32.0,
            estimated_duration_days=4.0,
            complexity_score=4.0,
            status=TaskStatus.COMPLETED,
            category="Game Logic",
            tags={"game-loop", "scenarios", "fsm"},
            acceptance_criteria=[
                "Phase-based FSM implemented",
                "Scenario configuration system working",
                "Objective tracking operational",
                "Phase transitions tested"
            ]
        ),
        
        TaskDefinition(
            task_id="task_11",
            name="Implement dynamic scoring and evaluation engine",
            description="Create weighted scoring, real-time calculation, performance tracking",
            dependencies=["task_10"],
            priority=TaskPriority.MUST,
            delivery_phase=DeliveryPhase.ALPHA,
            estimated_effort_hours=24.0,
            estimated_duration_days=3.0,
            complexity_score=3.0,
            status=TaskStatus.COMPLETED,
            category="Evaluation",
            tags={"scoring", "evaluation", "metrics"},
            acceptance_criteria=[
                "Weighted scoring system working",
                "Real-time score calculation operational",
                "Performance tracking implemented",
                "Comparative analysis functional"
            ]
        ),
        
        # Critical completion tasks for MVP
        TaskDefinition(
            task_id="task_43",
            name="Implement comprehensive evaluation and scoring metrics",
            description="Create attack/defense success tracking, time-to-mitigation metrics",
            dependencies=["task_11", "task_41"],
            priority=TaskPriority.MUST,
            delivery_phase=DeliveryPhase.ALPHA,
            estimated_effort_hours=24.0,
            estimated_duration_days=3.0,
            complexity_score=3.0,
            status=TaskStatus.NOT_STARTED,
            category="Metrics",
            tags={"evaluation", "metrics", "analytics"},
            acceptance_criteria=[
                "Attack success rate tracking implemented",
                "Defense success rate monitoring working",
                "Time-to-mitigation metrics operational",
                "Statistical analysis functional"
            ]
        ),
        
        TaskDefinition(
            task_id="task_45",
            name="Create comprehensive logging, debugging, and testing infrastructure",
            description="Implement detailed logging, debugging tools, comprehensive test suite",
            dependencies=["task_01", "task_02"],
            priority=TaskPriority.MUST,
            delivery_phase=DeliveryPhase.FOUNDATION,
            estimated_effort_hours=32.0,
            estimated_duration_days=4.0,
            complexity_score=3.0,
            status=TaskStatus.COMPLETED,
            category="Testing",
            tags={"logging", "debugging", "testing"},
            acceptance_criteria=[
                "Detailed logging implemented",
                "Debugging tools operational",
                "Comprehensive test suite passing",
                "Performance benchmarks established"
            ]
        ),
        
        # Additional important tasks for full system
        TaskDefinition(
            task_id="task_08",
            name="Implement LLM reasoning and behavior tree integration",
            description="Create LLM interface, behavior trees, GOAP/PDDL planning",
            dependencies=["task_01"],
            priority=TaskPriority.SHOULD,
            delivery_phase=DeliveryPhase.BETA,
            estimated_effort_hours=40.0,
            estimated_duration_days=5.0,
            complexity_score=4.0,
            status=TaskStatus.COMPLETED,
            category="AI",
            tags={"llm", "reasoning", "planning"},
            acceptance_criteria=[
                "LLM interface layer working",
                "Behavior trees implemented",
                "GOAP/PDDL planning operational",
                "Response validation functional"
            ]
        ),
        
        TaskDefinition(
            task_id="task_12",
            name="Build adversarial self-play and learning systems",
            description="Implement self-play mode, reinforcement learning, experience replay",
            dependencies=["task_08", "task_41"],
            priority=TaskPriority.SHOULD,
            delivery_phase=DeliveryPhase.BETA,
            estimated_effort_hours=48.0,
            estimated_duration_days=6.0,
            complexity_score=5.0,
            status=TaskStatus.COMPLETED,
            category="Learning",
            tags={"self-play", "rl", "learning"},
            acceptance_criteria=[
                "Self-play mode operational",
                "RL integration working",
                "Experience replay implemented",
                "Learning effectiveness measured"
            ]
        ),
        
        TaskDefinition(
            task_id="task_15",
            name="Build comprehensive monitoring and alerting infrastructure",
            description="Deploy Grafana dashboards, Prometheus metrics, alerting rules",
            dependencies=["task_05"],
            priority=TaskPriority.SHOULD,
            delivery_phase=DeliveryPhase.BETA,
            estimated_effort_hours=24.0,
            estimated_duration_days=3.0,
            complexity_score=3.0,
            status=TaskStatus.COMPLETED,
            category="Monitoring",
            tags={"monitoring", "alerting", "metrics"},
            acceptance_criteria=[
                "Grafana dashboards deployed",
                "Prometheus metrics collection working",
                "Alerting rules configured",
                "System health monitoring operational"
            ]
        )
    ]


def main():
    """Main function for critical path analysis"""
    try:
        # Initialize analyzer
        analyzer = CriticalPathAnalyzer()
        
        # Load Archangel project tasks
        tasks = create_archangel_task_definitions()
        for task in tasks:
            analyzer.add_task(task)
        
        print("Archangel Critical Path Analysis")
        print("=" * 50)
        
        # Analyze critical paths
        critical_paths = analyzer.analyze_critical_paths()
        print(f"\nIdentified {len(critical_paths)} paths, {sum(p.is_critical for p in critical_paths)} critical")
        
        # Show top 3 critical paths
        print("\nTop Critical Paths:")
        for i, path in enumerate(critical_paths[:3], 1):
            print(f"\n{i}. {path.path_id} ({'CRITICAL' if path.is_critical else 'Non-critical'})")
            print(f"   Duration: {path.total_duration:.1f} days")
            print(f"   Tasks: {' -> '.join(path.tasks)}")
            print(f"   Criticality Score: {path.criticality_score:.2f}")
        
        # Identify bottlenecks
        bottlenecks = analyzer.identify_bottlenecks()
        print(f"\nBottlenecks Identified:")
        print(f"- Dependency bottlenecks: {len(bottlenecks['dependency_bottlenecks'])}")
        print(f"- Task bottlenecks: {len(bottlenecks['task_bottlenecks'])}")
        print(f"- Risk bottlenecks: {len(bottlenecks['risk_bottlenecks'])}")
        
        # Define MVP
        core_features = [
            "Multi-agent coordination framework",
            "Basic Red vs Blue team agents",
            "Simple scenario execution",
            "Memory and knowledge systems",
            "Basic monitoring and logging"
        ]
        
        mvp = analyzer.define_mvp(core_features)
        print(f"\nMVP Definition:")
        print(f"- Required tasks: {len(mvp.required_tasks)}")
        print(f"- Estimated effort: {mvp.estimated_effort_hours:.1f} hours")
        print(f"- Estimated duration: {mvp.estimated_duration_days:.1f} days")
        
        # Optimize delivery sequence
        delivery_plan = analyzer.optimize_delivery_sequence()
        print(f"\nDelivery Plan:")
        for phase, tasks in delivery_plan.items():
            print(f"- {phase.value}: {len(tasks)} tasks")
        
        # Generate dashboard
        dashboard = analyzer.generate_project_dashboard()
        print(f"\nProject Overview:")
        print(f"- Total tasks: {dashboard['project_overview']['total_tasks']}")
        print(f"- Completed: {dashboard['project_overview']['completed_tasks']}")
        print(f"- In progress: {dashboard['project_overview']['in_progress_tasks']}")
        print(f"- Total effort: {dashboard['project_overview']['total_estimated_effort']:.1f}h")
        
        # Export analysis
        output_path = Path("project_analysis_output")
        analyzer.export_analysis(output_path)
        analyzer.visualize_dependency_graph(output_path)
        
        print(f"\nAnalysis exported to {output_path}/")
        print("Critical path analysis complete!")
        
    except Exception as e:
        logger.error(f"Critical path analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()