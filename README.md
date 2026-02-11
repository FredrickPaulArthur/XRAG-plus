# XRAG-plus

[![build-badge](https://img.shields.io/badge/build-pending-lightgrey)]()
[![docs-badge](https://img.shields.io/badge/docs-locally-blue)]()
[![license-badge](https://img.shields.io/badge/license-MIT-green)]()

**One-line elevator pitch**: XRAG+ is a multilingual retrieval-augmented generation system that ingests and indexes large-scale corpora (e.g., Wikipedia, CC-News), chunks the data, performs hybrid semantic–lexical retrieval with reranking, produces language-aware responses in open-world settings, and provides reproducible evaluation and ablation frameworks, building upon concepts introduced in [XRAG (arXiv)](https://doi.org/10.48550/arXiv.2505.10089).

## Description
XRAG+ is a multilingual Retrieval-Augmented Generation (RAG) system that I designed and implemented from scratch. It ingests large-scale multilingual corpora such as Wikipedia dumps, preprocesses them, and indexes them for hybrid retrieval. A Fusion-in-Decoder (FiD) generation module produces language-aware answers by fusing cross-lingual evidence. The project includes a complete evaluation framework, ablations, and reproducible scripts. XRAG+ demonstrates my ability to perform independent research, design AI systems, engineer scalable ML pipelines and conduct complex evaluations od AI systems.

Although benchmark datasets provide gold passages and distractors, these are insufficient to evaluate retrieval robustness in open-world settings. XRAG+ therefore operates over large-scale multilingual corpora (Wikipedia and CC-News), requiring the system to recover gold evidence from a realistic, noisy knowledge space.

---

## Table of contents
- [Quick links](#quick-links)
- [Current](#current)
- [High-level architecture](#high-level-architecture)
- [Get started](#get-started)
- [Modules](#modules)
- [License](#license)

## Quick links
- Install: [INSTALL.md](INSTALL.md)
- Quick demo: [QUICKSTART.md](QUICKSTART.md)
- Module READMEs: [src/README.md](src/README.md) and [src/*/README.md](src/)

## Current
- Platform: Windows
- Python: [3.13.2](https://www.python.org/downloads/release/python-3132/)
- Current branch: `master` (stable)

## High-level architecture
- XRAG-plus ingests documents → Chunking → Embeddings → Indexing → Retriever → Reranker → Summarizer → Generator.
- After Generation, the Evaluation and Ablations are carried out.

## Get started
For a quick demo that runs locally with a tiny example dataset, follow: [QUICKSTART.md](QUICKSTART.md).

## Modules
Top-level source is in `src/`. Each major area (chunking, indexing, retrieval, reranker, summarizer, generator) has its own README (`src/<module>/README.md`) explaining purpose, APIs, config keys and example snippets.

## License
This project is licensed under the MIT License — see [LICENSE](LICENSE).