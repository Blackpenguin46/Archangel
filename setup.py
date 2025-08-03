#!/usr/bin/env python3
"""
Archangel AI vs AI Cyber Conflict System - Setup Script
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="archangel",
    version="1.0.0",
    author="Blackpenguin46",
    description="AI vs AI Autonomous Cyber Conflict System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'archangel-red=run_red_agent:main',
            'archangel-blue=run_blue_agent:main', 
            'archangel-orchestrator=run_orchestrator:main',
            'archangel-demo=run_demo:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Information Technology",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="ai cybersecurity penetration-testing security-monitoring autonomous-agents",
    project_urls={
        "Bug Reports": "https://github.com/blackpenguin46/archangel/issues",
        "Source": "https://github.com/blackpenguin46/archangel",
        "Documentation": "https://archangel-ai.readthedocs.io/",
    },
)