# Grok-1 推荐系统学习指南

欢迎来到 Grok-1 推荐系统（X 算法）的学习指南。这里汇集了关于系统架构、核心算法和代码实现的详细文档。

## 目录

1.  **[系统概览](01_system_overview.md)**
    *   了解整个推荐系统的宏观架构，包括 Candidate Pipeline, Home Mixer 等核心组件。

2.  **[Home Mixer 逻辑](02_home_mixer_logic.md)**
    *   深入了解主页推荐的混合逻辑，如何从不同来源聚合内容。

3.  **[Phoenix 模型架构](03_phoenix_model.md)**
    *   解析基于 Transformer 的核心排序模型 Phoenix 的设计原理。

4.  **[打分与排序](04_scoring_and_ranking.md)**
    *   学习多目标打分机制以及最终的排序策略。

5.  **[过滤规则](05_filtering_rules.md)**
    *   了解系统如何过滤掉重复、已读或不适宜的内容。

6.  **[实战分析: run_ranker.py](06_run_ranker_analysis.md)** :new:
    *   **推荐阅读**：通过分析演示脚本，一步步理解模型如何加载、处理数据并生成预测结果。包含详细的 Mermaid 流程图和代码逻辑解析。

## 快速开始

如果你想动手运行代码，请参考以下步骤（Windows 环境）：

1.  确保安装了 Python 3.11+ (推荐使用 Miniconda)。
2.  安装 `uv` 包管理工具: `pip install uv`。
3.  进入 `phoenix` 目录。
4.  运行演示脚本:
    ```bash
    python -m uv run run_ranker.py
