# Grok-1 Recommendation System Learning Guide

Welcome to the learning guide for the Grok-1 Recommendation System (X Algorithm). This guide allows you to explore detailed documentation on system architecture, core algorithms, and code implementation.

## Table of Contents

1.  **[System Overview](01_system_overview.md)**
    *   Understand the macro architecture of the entire recommendation system, including core components like Candidate Pipeline and Home Mixer.

2.  **[Home Mixer Logic](02_home_mixer_logic.md)**
    *   Dive deep into the mixing logic of the home feed and how content is aggregated from different sources.

3.  **[Phoenix Model Architecture](03_phoenix_model.md)**
    *   Analyze the design principles of the core ranking model based on Transformer, Phoenix.

4.  **[Scoring and Ranking](04_scoring_and_ranking.md)**
    *   Learn about the multi-objective scoring mechanism and the final ranking strategy.

5.  **[Filtering Rules](05_filtering_rules.md)**
    *   Understand how the system filters out duplicate, read, or inappropriate content.

6.  **[Practical Analysis: run_ranker.py](06_run_ranker_analysis.md)** :new:
    *   **Recommended Reading**: Through analyzing the demo script, understand step-by-step how the model loads, processes data, and generates prediction results. Includes detailed Mermaid flowcharts and code logic parsing.

## Quick Start

If you want to run the code yourself, please refer to the following steps (Windows environment):

1.  Ensure Python 3.11+ is installed (Miniconda is recommended).
2.  Install the `uv` package management tool: `pip install uv`.
3.  Enter the `phoenix` directory.
4.  Run the demo script:
    ```bash
    python -m uv run run_ranker.py
    ```
