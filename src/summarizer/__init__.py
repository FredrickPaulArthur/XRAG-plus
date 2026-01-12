"""XRAG+ summarizer package exports.


Expose Settings and Summarizer for easy imports:
from src.summarizer import Summarizer, Settings

For working on this module,
pip install sentence-transformers==5.1.2 transformers==4.57.3 torch==2.9.1 openai==2.8.1 cohere==5.20.0 dotenv==0.9.9 spacy==3.8.11 hf_xet==1.2.
"""


from .config import Settings
from .summarizer import Summarizer
from .model_wrapper import HFWrapper, OpenAIWrapper, CohereAIWrapper
from .prompts import summ_prompt


__all__ = [
    "Settings",
    "Summarizer",
    "HFWrapper",
    "OpenAIWrapper",
    "CohereAIWrapper",
    "summ_prompt"
]