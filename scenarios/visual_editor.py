#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Visual Scenario Editor
Drag-and-drop interface for non-technical users to create scenarios
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkinter.scrolledtext as scrolledtext

from .scenario_dsl import ScenarioDSL, create_dsl_example
from .dsl_parser import DSLParser, ValidationLevel
from .scenario_templates import ScenarioType, ScenarioCategory, ComplexityLevel, NetworkTopology

logger = logging.getLogger(__name__)

class ComponentType(Enum):
    """Visual component types"""
    SCENARIO = "scenario"
    PARAMETER = "parameter"
    OBJECTIVE = "objective"
    ASSET = "asset"
    NETWORK = "network"
    TEAM = "team"
    VALIDATION = "validation"
    DOCUMENTATION = "documentation"

@dataclass
class VisualComponent:
    """Visual component for drag-and-drop interface"""
    component_id: str
    component_type: ComponentType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    position: Tuple[int, int] = (0, 0)
    connections: List[str] = field(default_factory=list)
    
    # Visual properties
    color: str = "#lightblue"
    width: int = 150
    height: int = 80
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)

class VisualScenarioEditor:
    """
    Visual drag-and-drop scenario editor for non-technical users.
    
    Features:
    - Drag-and-drop component placement
    - Property editing panels
    - Real-time DSL code generation
    - Visual validation feedback
    - Template library integration
    - Export to DSL and JSON formats
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Archangel Scenario Editor")
        self.root.geometry("1200x800")
        
        # Editor state
        self.components: Dict[str, VisualComponent] = {}
        self.selected_component: Optional[str] = None
        self.canvas_offset = (0, 0)
        self.zoom_level = 1.0
        
        # DSL integration
        self.dsl_parser = DSLParser()
        self.current_dsl_code = ""
        
        # UI components
        self.canvas = None
        self.property_panel = None
        self.component_palette = None
        self.dsl_preview = None
        self.validation_panel = None
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize UI
        self._create_ui()
        self._create_component_templates()
    
    def run(self):
        """Start the visual editor"""
        self.root.mainloop()
    
    def _create_ui(self):
        """Create the main UI layout"""
        # Create main panes
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Component palette and properties
        left_panel = ttk.Frame(main_paned)
        main_paned.add(left_panel, weight=1)
        
        # Center panel - Canvas
        center_panel = ttk.Frame(main_paned)
        main_paned.add(center_panel, weight=3)
        
        # Right panel - DSL preview and validation
        right_panel = ttk.Frame(main_paned)
        main_paned.add(right_panel, weight=2)
        
        # Create left panel components
        self._create_component_palette(left_panel)
        self._create_property_panel(left_panel)
        
        # Create center panel - canvas
        self._create_canvas(center_panel)
        
        # Create right panel components
        self._create_dsl_preview(right_panel)
        self._create_validation_panel(right_panel)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create toolbar
        self._create_toolbar()
    
    def _create_menu_bar(self):
        """Create the menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self._new_scenario)
        file_menu.add_command(label="Open", command=self._open_scenario)
        file_menu.add_command(label="Save", command=self._save_scenario)
        file_menu.add_command(label="Save As", command=self._save_scenario_as)
        file_menu.add_separator()
        file_menu.add_command(label="Export DSL", command=self._export_dsl)
        file_menu.add_command(label="Export JSON", command=self._export_json)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self._undo)
        edit_menu.add_command(label="Redo", command=self._redo)
        edit_menu.add_separator()
        edit_menu.add_command(label="Copy", command=self._copy_component)
        edit_menu.add_command(label="Paste", command=self._paste_component)
        edit_menu.add_command(label="Delete", command=self._delete_component)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Zoom In", command=self._zoom_in)
        view_menu.add_command(label="Zoom Out", command=self._zoom_out)
        view_menu.add_command(label="Reset Zoom", command=self._reset_zoom)
        view_menu.add_separator()
        view_menu.add_command(label="Show Grid", command=self._toggle_grid)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Validate", command=self._validate_scenario)
        tools_menu.add_command(label="Generate DSL", command=self._generate_dsl)
        tools_menu.add_command(label="Load Example", command=self._load_example)
    
    def _create_toolbar(self):
        """Create the toolbar"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        # Toolbar buttons
        ttk.Button(toolbar, text="New", command=self._new_scenario).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Open", command=self._open_scenario).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save", command=self._save_scenario).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        ttk.Button(toolbar, text="Validate", command=self._validate_scenario).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Generate DSL", command=self._generate_dsl).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        ttk.Button(toolbar, text="Zoom In", command=self._zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Zoom Out", command=self._zoom_out).pack(side=tk.LEFT, padx=2)
    
    def _create_component_palette(self, parent):
        """Create the component palette"""
        palette_frame = ttk.LabelFrame(parent, text="Components")
        palette_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Component buttons
        components = [
            ("Scenario", ComponentType.SCENARIO, "#FFE4B5"),
            ("Parameter", ComponentType.PARAMETER, "#E0E0E0"),
            ("Objective", ComponentType.OBJECTIVE, "#98FB98"),
            ("Asset", ComponentType.ASSET, "#87CEEB"),
            ("Network", ComponentType.NETWORK, "#DDA0DD"),
            ("Team", ComponentType.TEAM, "#F0E68C"),
            ("Validation", ComponentType.VALIDATION, "#FFA07A"),
            ("Documentation", ComponentType.DOCUMENTATION, "#D3D3D3")
        ]
        
        for name, comp_type, color in components:
            btn = tk.Button(
                palette_frame,
                text=name,
                bg=color,
                command=lambda t=comp_type: self._add_component(t)
            )
            btn.pack(fill=tk.X, padx=5, pady=2)
    
    def _create_property_panel(self, parent):
        """Create the property editing panel"""
        prop_frame = ttk.LabelFrame(parent, text="Properties")
        prop_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollable frame for properties
        canvas = tk.Canvas(prop_frame)
        scrollbar = ttk.Scrollbar(prop_frame, orient="vertical", command=canvas.yview)
        self.property_panel = ttk.Frame(canvas)
        
        self.property_panel.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.property_panel, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Default message
        ttk.Label(self.property_panel, text="Select a component to edit properties").pack(pady=20)
    
    def _create_canvas(self, parent):
        """Create the main canvas for component placement"""
        canvas_frame = ttk.LabelFrame(parent, text="Scenario Canvas")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas with scrollbars
        canvas_container = ttk.Frame(canvas_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(
            canvas_container,
            bg="white",
            scrollregion=(0, 0, 2000, 2000)
        )
        
        h_scrollbar = ttk.Scrollbar(canvas_container, orient="horizontal", command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=self.canvas.yview)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        self.canvas.grid(row=0, column=0, sticky="nsew")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        
        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self._canvas_click)
        self.canvas.bind("<B1-Motion>", self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)
        self.canvas.bind("<Double-Button-1>", self._canvas_double_click)
    
    def _create_dsl_preview(self, parent):
        """Create the DSL code preview panel"""
        dsl_frame = ttk.LabelFrame(parent, text="Generated DSL Code")
        dsl_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # DSL text area
        self.dsl_preview = scrolledtext.ScrolledText(
            dsl_frame,
            wrap=tk.WORD,
            height=15,
            font=("Courier", 10)
        )
        self.dsl_preview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # DSL buttons
        dsl_buttons = ttk.Frame(dsl_frame)
        dsl_buttons.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(dsl_buttons, text="Generate", command=self._generate_dsl).pack(side=tk.LEFT, padx=2)
        ttk.Button(dsl_buttons, text="Copy", command=self._copy_dsl).pack(side=tk.LEFT, padx=2)
        ttk.Button(dsl_buttons, text="Save", command=self._save_dsl).pack(side=tk.LEFT, padx=2)
    
    def _create_validation_panel(self, parent):
        """Create the validation results panel"""
        val_frame = ttk.LabelFrame(parent, text="Validation Results")
        val_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Validation text area
        self.validation_panel = scrolledtext.ScrolledText(
            val_frame,
            wrap=tk.WORD,
            height=10,
            font=("Courier", 9)
        )
        self.validation_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Validation buttons
        val_buttons = ttk.Frame(val_frame)
        val_buttons.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(val_buttons, text="Validate", command=self._validate_scenario).pack(side=tk.LEFT, padx=2)
        ttk.Button(val_buttons, text="Clear", command=self._clear_validation).pack(side=tk.LEFT, padx=2)
    
    def _create_component_templates(self):
        """Create templates for different component types"""
        self.component_templates = {
            ComponentType.SCENARIO: {
                "name": "New Scenario",
                "properties": {
                    "description": "",
                    "type": "TRAINING",
                    "category": "FULL_CAMPAIGN",
                    "complexity": "INTERMEDIATE",
                    "duration_hours": 2,
                    "min_participants": 1,
                    "max_participants": 10
                },
                "color": "#FFE4B5"
            },
            ComponentType.PARAMETER: {
                "name": "New Parameter",
                "properties": {
                    "param_type": "string",
                    "default_value": "",
                    "required": True,
                    "description": ""
                },
                "color": "#E0E0E0"
            },
            ComponentType.OBJECTIVE: {
                "name": "New Objective",
                "properties": {
                    "description": "",
                    "type": "primary",
                    "points": 100,
                    "time_limit_minutes": 0,
                    "success_criteria": []
                },
                "color": "#98FB98"
            },
            ComponentType.ASSET: {
                "name": "New Asset",
                "properties": {
                    "asset_type": "vm",
                    "configuration": {},
                    "vulnerabilities": [],
                    "security_controls": []
                },
                "color": "#87CEEB"
            },
            ComponentType.NETWORK: {
                "name": "Network Config",
                "properties": {
                    "topology": "SIMPLE_NETWORK"
                },
                "color": "#DDA0DD"
            },
            ComponentType.TEAM: {
                "name": "Team Config",
                "properties": {
                    "teams": ["RED_TEAM", "BLUE_TEAM"]
                },
                "color": "#F0E68C"
            },
            ComponentType.VALIDATION: {
                "name": "Validation Rule",
                "properties": {
                    "rule": ""
                },
                "color": "#FFA07A"
            },
            ComponentType.DOCUMENTATION: {
                "name": "Documentation",
                "properties": {
                    "content": ""
                },
                "color": "#D3D3D3"
            }
        }
    
    def _add_component(self, component_type: ComponentType):
        """Add a new component to the canvas"""
        try:
            template = self.component_templates[component_type]
            
            component = VisualComponent(
                component_id=str(uuid.uuid4()),
                component_type=component_type,
                name=template["name"],
                properties=template["properties"].copy(),
                position=(100, 100),
                color=template["color"]
            )
            
            self.components[component.component_id] = component
            self._draw_component(component)
            self._generate_dsl()
            
        except Exception as e:
            self.logger.error(f"Failed to add component: {e}")
            messagebox.showerror("Error", f"Failed to add component: {str(e)}")
    
    def _draw_component(self, component: VisualComponent):
        """Draw a component on the canvas"""
        try:
            x, y = component.position
            
            # Draw component rectangle
            rect_id = self.canvas.create_rectangle(
                x, y, x + component.width, y + component.height,
                fill=component.color,
                outline="black",
                width=2,
                tags=(component.component_id, "component")
            )
            
            # Draw component text
            text_id = self.canvas.create_text(
                x + component.width // 2,
                y + component.height // 2,
                text=component.name,
                font=("Arial", 10, "bold"),
                tags=(component.component_id, "component_text")
            )
            
            # Store canvas item IDs
            component.properties["_canvas_rect"] = rect_id
            component.properties["_canvas_text"] = text_id
            
        except Exception as e:
            self.logger.error(f"Failed to draw component: {e}")
    
    def _canvas_click(self, event):
        """Handle canvas click events"""
        try:
            # Find clicked component
            clicked_items = self.canvas.find_overlapping(event.x, event.y, event.x, event.y)
            
            component_id = None
            for item in clicked_items:
                tags = self.canvas.gettags(item)
                for tag in tags:
                    if tag in self.components:
                        component_id = tag
                        break
                if component_id:
                    break
            
            if component_id:
                self._select_component(component_id)
            else:
                self._deselect_component()
                
        except Exception as e:
            self.logger.error(f"Canvas click error: {e}")
    
    def _canvas_drag(self, event):
        """Handle canvas drag events"""
        if self.selected_component:
            try:
                component = self.components[self.selected_component]
                
                # Update component position
                new_x = max(0, event.x - component.width // 2)
                new_y = max(0, event.y - component.height // 2)
                component.position = (new_x, new_y)
                
                # Move canvas items
                rect_id = component.properties.get("_canvas_rect")
                text_id = component.properties.get("_canvas_text")
                
                if rect_id:
                    self.canvas.coords(rect_id, new_x, new_y, new_x + component.width, new_y + component.height)
                if text_id:
                    self.canvas.coords(text_id, new_x + component.width // 2, new_y + component.height // 2)
                    
            except Exception as e:
                self.logger.error(f"Canvas drag error: {e}")
    
    def _canvas_release(self, event):
        """Handle canvas release events"""
        pass
    
    def _canvas_double_click(self, event):
        """Handle canvas double-click events"""
        # Double-click to edit component properties
        self._canvas_click(event)
        if self.selected_component:
            self._edit_component_properties()
    
    def _select_component(self, component_id: str):
        """Select a component and show its properties"""
        try:
            # Deselect previous component
            if self.selected_component:
                self._highlight_component(self.selected_component, False)
            
            # Select new component
            self.selected_component = component_id
            self._highlight_component(component_id, True)
            self._show_component_properties(component_id)
            
        except Exception as e:
            self.logger.error(f"Component selection error: {e}")
    
    def _deselect_component(self):
        """Deselect the current component"""
        if self.selected_component:
            self._highlight_component(self.selected_component, False)
            self.selected_component = None
            self._clear_property_panel()
    
    def _highlight_component(self, component_id: str, highlight: bool):
        """Highlight or unhighlight a component"""
        try:
            component = self.components[component_id]
            rect_id = component.properties.get("_canvas_rect")
            
            if rect_id:
                if highlight:
                    self.canvas.itemconfig(rect_id, outline="red", width=3)
                else:
                    self.canvas.itemconfig(rect_id, outline="black", width=2)
                    
        except Exception as e:
            self.logger.error(f"Component highlight error: {e}")
    
    def _show_component_properties(self, component_id: str):
        """Show component properties in the property panel"""
        try:
            component = self.components[component_id]
            
            # Clear property panel
            for widget in self.property_panel.winfo_children():
                widget.destroy()
            
            # Component header
            header_frame = ttk.Frame(self.property_panel)
            header_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(header_frame, text=f"{component.component_type.value.title()}", font=("Arial", 12, "bold")).pack()
            
            # Name field
            name_frame = ttk.Frame(self.property_panel)
            name_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(name_frame, text="Name:").pack(side=tk.LEFT)
            name_var = tk.StringVar(value=component.name)
            name_entry = ttk.Entry(name_frame, textvariable=name_var)
            name_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            name_var.trace('w', lambda *args: self._update_component_name(component_id, name_var.get()))
            
            # Properties
            for prop_name, prop_value in component.properties.items():
                if prop_name.startswith('_'):  # Skip internal properties
                    continue
                
                prop_frame = ttk.Frame(self.property_panel)
                prop_frame.pack(fill=tk.X, padx=5, pady=2)
                
                ttk.Label(prop_frame, text=f"{prop_name}:").pack(side=tk.LEFT)
                
                if isinstance(prop_value, bool):
                    var = tk.BooleanVar(value=prop_value)
                    widget = ttk.Checkbutton(prop_frame, variable=var)
                    var.trace('w', lambda *args, p=prop_name, v=var: self._update_component_property(component_id, p, v.get()))
                elif isinstance(prop_value, (int, float)):
                    var = tk.StringVar(value=str(prop_value))
                    widget = ttk.Entry(prop_frame, textvariable=var)
                    var.trace('w', lambda *args, p=prop_name, v=var: self._update_component_property(component_id, p, self._convert_value(v.get(), type(prop_value))))
                elif isinstance(prop_value, list):
                    var = tk.StringVar(value=', '.join(map(str, prop_value)))
                    widget = ttk.Entry(prop_frame, textvariable=var)
                    var.trace('w', lambda *args, p=prop_name, v=var: self._update_component_property(component_id, p, [x.strip() for x in v.get().split(',') if x.strip()]))
                else:
                    var = tk.StringVar(value=str(prop_value))
                    widget = ttk.Entry(prop_frame, textvariable=var)
                    var.trace('w', lambda *args, p=prop_name, v=var: self._update_component_property(component_id, p, v.get()))
                
                widget.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            
            # Action buttons
            button_frame = ttk.Frame(self.property_panel)
            button_frame.pack(fill=tk.X, padx=5, pady=10)
            
            ttk.Button(button_frame, text="Delete", command=lambda: self._delete_component_by_id(component_id)).pack(side=tk.LEFT, padx=2)
            ttk.Button(button_frame, text="Duplicate", command=lambda: self._duplicate_component(component_id)).pack(side=tk.LEFT, padx=2)
            
        except Exception as e:
            self.logger.error(f"Property panel error: {e}")
    
    def _clear_property_panel(self):
        """Clear the property panel"""
        for widget in self.property_panel.winfo_children():
            widget.destroy()
        ttk.Label(self.property_panel, text="Select a component to edit properties").pack(pady=20)
    
    def _update_component_name(self, component_id: str, new_name: str):
        """Update component name"""
        try:
            component = self.components[component_id]
            component.name = new_name
            component.modified_at = datetime.now()
            
            # Update canvas text
            text_id = component.properties.get("_canvas_text")
            if text_id:
                self.canvas.itemconfig(text_id, text=new_name)
            
            self._generate_dsl()
            
        except Exception as e:
            self.logger.error(f"Name update error: {e}")
    
    def _update_component_property(self, component_id: str, prop_name: str, new_value: Any):
        """Update component property"""
        try:
            component = self.components[component_id]
            component.properties[prop_name] = new_value
            component.modified_at = datetime.now()
            
            self._generate_dsl()
            
        except Exception as e:
            self.logger.error(f"Property update error: {e}")
    
    def _convert_value(self, value_str: str, target_type: type):
        """Convert string value to target type"""
        try:
            if target_type == int:
                return int(value_str)
            elif target_type == float:
                return float(value_str)
            else:
                return value_str
        except ValueError:
            return 0 if target_type in [int, float] else ""
    
    def _generate_dsl(self):
        """Generate DSL code from visual components"""
        try:
            dsl_lines = []
            
            # Find scenario component
            scenario_component = None
            for component in self.components.values():
                if component.component_type == ComponentType.SCENARIO:
                    scenario_component = component
                    break
            
            if scenario_component:
                # Generate scenario definition
                props = scenario_component.properties
                dsl_lines.append(f'scenario("{scenario_component.name}",')
                dsl_lines.append(f'    type=ScenarioType.{props.get("type", "TRAINING")},')
                dsl_lines.append(f'    category=ScenarioCategory.{props.get("category", "FULL_CAMPAIGN")},')
                dsl_lines.append(f'    complexity=ComplexityLevel.{props.get("complexity", "INTERMEDIATE")},')
                dsl_lines.append(f'    duration=timedelta(hours={props.get("duration_hours", 2)})')
                dsl_lines.append(')')
                dsl_lines.append('')
                
                if props.get("description"):
                    dsl_lines.append(f'description("""{props["description"]}""")')
                    dsl_lines.append('')
            
            # Generate other components
            for component in self.components.values():
                if component.component_type == ComponentType.SCENARIO:
                    continue
                
                if component.component_type == ComponentType.PARAMETER:
                    props = component.properties
                    dsl_lines.append(f'parameter("{component.name}", "{props.get("param_type", "string")}",')
                    dsl_lines.append(f'    default={repr(props.get("default_value", ""))},')
                    dsl_lines.append(f'    required={props.get("required", True)},')
                    if props.get("description"):
                        dsl_lines.append(f'    description="{props["description"]}"')
                    dsl_lines.append(')')
                    dsl_lines.append('')
                
                elif component.component_type == ComponentType.OBJECTIVE:
                    props = component.properties
                    dsl_lines.append(f'objective("{component.name}",')
                    dsl_lines.append(f'    "{props.get("description", "")}",')
                    dsl_lines.append(f'    type="{props.get("type", "primary")}",')
                    dsl_lines.append(f'    points={props.get("points", 100)}')
                    if props.get("time_limit_minutes", 0) > 0:
                        dsl_lines.append(f'    time_limit=timedelta(minutes={props["time_limit_minutes"]})')
                    dsl_lines.append(')')
                    dsl_lines.append('')
                
                elif component.component_type == ComponentType.ASSET:
                    props = component.properties
                    dsl_lines.append(f'asset("{component.name}", "{props.get("asset_type", "vm")}",')
                    if props.get("configuration"):
                        dsl_lines.append(f'    configuration={props["configuration"]},')
                    if props.get("vulnerabilities"):
                        dsl_lines.append(f'    vulnerabilities={props["vulnerabilities"]},')
                    if props.get("security_controls"):
                        dsl_lines.append(f'    security_controls={props["security_controls"]}')
                    dsl_lines.append(')')
                    dsl_lines.append('')
            
            self.current_dsl_code = '\n'.join(dsl_lines)
            
            # Update DSL preview
            self.dsl_preview.delete(1.0, tk.END)
            self.dsl_preview.insert(1.0, self.current_dsl_code)
            
        except Exception as e:
            self.logger.error(f"DSL generation error: {e}")
            self.current_dsl_code = f"# Error generating DSL: {str(e)}"
            self.dsl_preview.delete(1.0, tk.END)
            self.dsl_preview.insert(1.0, self.current_dsl_code)
    
    def _validate_scenario(self):
        """Validate the current scenario"""
        try:
            if not self.current_dsl_code.strip():
                self._generate_dsl()
            
            if not self.current_dsl_code.strip():
                messagebox.showwarning("Warning", "No scenario to validate")
                return
            
            # Validate using DSL parser
            validation_result = self.dsl_parser.validate(self.current_dsl_code, ValidationLevel.COMPREHENSIVE)
            
            # Display results
            self.validation_panel.delete(1.0, tk.END)
            
            if validation_result.valid:
                self.validation_panel.insert(tk.END, "✓ Validation PASSED\n\n", "success")
            else:
                self.validation_panel.insert(tk.END, "✗ Validation FAILED\n\n", "error")
            
            if validation_result.syntax_errors:
                self.validation_panel.insert(tk.END, "Syntax Errors:\n")
                for error in validation_result.syntax_errors:
                    self.validation_panel.insert(tk.END, f"  • {error}\n")
                self.validation_panel.insert(tk.END, "\n")
            
            if validation_result.semantic_errors:
                self.validation_panel.insert(tk.END, "Semantic Errors:\n")
                for error in validation_result.semantic_errors:
                    self.validation_panel.insert(tk.END, f"  • {error}\n")
                self.validation_panel.insert(tk.END, "\n")
            
            if validation_result.warnings:
                self.validation_panel.insert(tk.END, "Warnings:\n")
                for warning in validation_result.warnings:
                    self.validation_panel.insert(tk.END, f"  • {warning}\n")
                self.validation_panel.insert(tk.END, "\n")
            
            if validation_result.suggestions:
                self.validation_panel.insert(tk.END, "Suggestions:\n")
                for suggestion in validation_result.suggestions:
                    self.validation_panel.insert(tk.END, f"  • {suggestion}\n")
                self.validation_panel.insert(tk.END, "\n")
            
            # Performance info
            self.validation_panel.insert(tk.END, f"Validation completed in {validation_result.validation_time:.3f}s\n")
            self.validation_panel.insert(tk.END, f"Complexity score: {validation_result.complexity_score:.1f}/100\n")
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            messagebox.showerror("Error", f"Validation failed: {str(e)}")
    
    # Placeholder methods for menu actions
    def _new_scenario(self): pass
    def _open_scenario(self): pass
    def _save_scenario(self): pass
    def _save_scenario_as(self): pass
    def _export_dsl(self): pass
    def _export_json(self): pass
    def _undo(self): pass
    def _redo(self): pass
    def _copy_component(self): pass
    def _paste_component(self): pass
    def _delete_component(self): pass
    def _zoom_in(self): pass
    def _zoom_out(self): pass
    def _reset_zoom(self): pass
    def _toggle_grid(self): pass
    def _copy_dsl(self): pass
    def _save_dsl(self): pass
    def _clear_validation(self): pass
    def _edit_component_properties(self): pass
    def _delete_component_by_id(self, component_id: str): pass
    def _duplicate_component(self, component_id: str): pass
    def _load_example(self):
        """Load an example scenario"""
        try:
            # Clear current components
            self.components.clear()
            self.canvas.delete("all")
            
            # Add example components
            scenario_comp = VisualComponent(
                component_id=str(uuid.uuid4()),
                component_type=ComponentType.SCENARIO,
                name="Example APT Simulation",
                properties={
                    "description": "Advanced Persistent Threat simulation exercise",
                    "type": "LIVE_EXERCISE",
                    "category": "FULL_CAMPAIGN",
                    "complexity": "ADVANCED",
                    "duration_hours": 4
                },
                position=(50, 50),
                color="#FFE4B5"
            )
            
            self.components[scenario_comp.component_id] = scenario_comp
            self._draw_component(scenario_comp)
            
            # Add objective
            obj_comp = VisualComponent(
                component_id=str(uuid.uuid4()),
                component_type=ComponentType.OBJECTIVE,
                name="Initial Access",
                properties={
                    "description": "Gain initial foothold in target network",
                    "type": "primary",
                    "points": 100,
                    "time_limit_minutes": 60
                },
                position=(250, 50),
                color="#98FB98"
            )
            
            self.components[obj_comp.component_id] = obj_comp
            self._draw_component(obj_comp)
            
            # Generate DSL
            self._generate_dsl()
            
            messagebox.showinfo("Success", "Example scenario loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to load example: {e}")
            messagebox.showerror("Error", f"Failed to load example: {str(e)}")

def main():
    """Main function to run the visual editor"""
    try:
        editor = VisualScenarioEditor()
        editor.run()
    except Exception as e:
        logger.error(f"Failed to start visual editor: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()