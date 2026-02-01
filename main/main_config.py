from dataclasses import dataclass, field
from typing import Dict, Any

from src.chunker.config import Settings as ChunkerSettings
from src.indexing.config import Settings as IndexerSettings
from src.retrieval.config import Settings as RetrievalSettings
from src.reranker.config import Settings as RerankerSettings
from src.generator.config import Settings as GeneratorSettings
from src.summarizer.config import Settings as SummarizerSettings



@dataclass
class MainConfig:
    """
    XRAG+ Global Configuration Root.

    - Single source of truth
    - Controls ablations, experiments, pipelines
    """

    chunker: ChunkerSettings = field(default_factory=ChunkerSettings)
    indexer: IndexerSettings = field(default_factory=IndexerSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    reranker: RerankerSettings = field(default_factory=RerankerSettings)
    generator: GeneratorSettings = field(default_factory=GeneratorSettings)
    summarizer: SummarizerSettings = field(default_factory=SummarizerSettings)


    # Global experiment flags
    ENABLE_RERANKING: bool = True
    ENABLE_SUMMARIZATION: bool = True
    ENABLE_DEDUP: bool = True

    EXPERIMENT_NAME: str = "xrag_default"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunker": self.chunker.__dict__,
            "indexing": self.indexer.__dict__,
            "retrieval": self.retrieval.__dict__,
            "reranker": self.reranker.__dict__,
            "generator": self.generator.__dict__,
            "summarizer": self.summarizer.__dict__,
            "flags": {
                "ENABLE_RERANKING": self.ENABLE_RERANKING,
                "ENABLE_SUMMARIZATION": self.ENABLE_SUMMARIZATION,
                "ENABLE_DEDUP": self.ENABLE_DEDUP,
            },
            "experiment": self.EXPERIMENT_NAME,
        }