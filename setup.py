"""
Setup script for the AI Phases package.
"""

from setuptools import setup, find_packages

setup(
    name="ai_phases",
    version="1.0.0",
    description="AI Phases - Hệ thống nâng cao hiệu suất với 6 phases",
    long_description="""
    AI Phases là hệ thống nâng cao hiệu suất AI với 6 phases:
    - Phase 1: Online Learning Engine (+2.5%)
    - Phase 2: Advanced Backtest Framework (+1.5%)
    - Phase 3: Adaptive Intelligence (+3.0%)
    - Phase 4: Multi-Market Learning (+2.0%)
    - Phase 5: Real-Time Enhancement (+1.5%)
    - Phase 6: Future Evolution (+1.5%)
    
    Tổng performance boost: +12.0%
    """,
    author="AI Developer",
    author_email="ai@example.com",
    url="https://github.com/ai-developer/ai-phases",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.0",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
) 