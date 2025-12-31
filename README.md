# XRAG-plus
XRAG+ : Hybrid Retrieval and Tool-Augmented Verification for Robust Cross-Lingual Generation

## Description
XRAG+ is a multilingual Retrieval-Augmented Generation (RAG) system that I designed and implemented from scratch. It ingests large-scale multilingual corpora such as Wikipedia dumps, preprocesses them, and indexes them for hybrid retrieval. A Fusion-in-Decoder (FiD) generation module produces language-aware answers by fusing cross-lingual evidence. The project includes a complete evaluation framework, ablations, and reproducible scripts. XRAG+ demonstrates my ability to perform independent research, design complex AI systems, and engineer scalable ML pipelines.

Although benchmark datasets provide gold passages and distractors, these are insufficient to evaluate retrieval robustness in open-world settings. XRAG+ therefore operates over large-scale multilingual corpora (Wikipedia and CC-News), requiring the system to recover gold evidence from a realistic, noisy knowledge space.