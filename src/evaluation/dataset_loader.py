# src/evaluation/dataset_loader.py
"""
DatasetLoader - normalize multiple QA dataset formats into a single QAExample schema.

Supported datasets (by dataset_name):
 - "mlqa"   : expects a SQuAD-like JSON or per-language JSON produced by MLQA extraction
 - "mkqa"   : expects mkqa.jsonl (each line is JSON with 'query', 'answers', 'queries', etc.)
 - "tydiqa" : TyDiQA JSON format (articles -> paragraphs -> qas)
 - "xquad"  : SQuAD-style JSON
 - "ccnews"/"wiki": Generic corpora (returns docs as entries with no question by default)

Unified output example:
{
  "id": "unique-id",
  "question": "text",
  "answers": ["ans1", "ans2"],
  "context": "the passage text",
  "lang": "en",
  "relevant_doc_ids": []
}
"""


from typing import List, Dict, Any, Optional
import os, warnings
from src.evaluation.utils import _read_json, _read_jsonl



class DatasetLoader:
    """
    Load and normalize multilingual QA datasets.

    Args:
      dataset_name  : one of `['mlqa', 'mkqa', 'tydiqa', 'xquad', 'ccnews', 'wiki']`
      file_path     : path to the dataset file or directory. If directory, loader will try to find language-specific files.
      lang          : optional language code filter `(e.g., 'en','hi','de','es','ru')`. For MKQA, selects the language variant.
      max_examples  : optional cap for memory-limited quick tests.
    """

    def __init__(self, dataset_name: str, file_path: str, lang: Optional[str] = None, max_examples: Optional[int] = None):
        self.dataset_name = dataset_name.lower()
        self.file_path = file_path
        self.lang = lang
        self.max_examples = max_examples


    def load(self) -> List[Dict[str, Any]]:
        """
        Entry Point for Dataset Loading.
        ....

        Load and normalize into list of examples (canonical schema).
        """
        if self.dataset_name == "mlqa":
            return self._load_mlqa(self.file_path, lang=self.lang)
        elif self.dataset_name == "mkqa":
            return self._load_mkqa_jsonl(self.file_path, lang=self.lang)
        elif self.dataset_name == "tydiqa":
            return self._load_tydiqa(self.file_path, lang=self.lang)
        elif self.dataset_name == "xquad" or self.dataset_name == "squad":
            return self._load_squad(self.file_path, lang=self.lang)

        elif self.dataset_name in ("ccnews", "wiki", "corpus"):
            return self._load_corpus(self.file_path, max_examples=self.max_examples, lang=self.lang)


    # ---------------------------
    # Format-specific loaders
    # ---------------------------

    def _load_mlqa(self, path: str, lang: Optional[str] = "en") -> List[Dict[str, Any]]:
        """
        Load MLQA-style JSON.
        ...
        - MLQA sometimes provides per-language files with SQuAD-like structure,

        {
          'version':...,
          'data': [
            {
              'title':...,
              'paragraphs':[
                {
                  'context':...,
                  'qas':[
                    {
                      id, question,
                      answers:[
                        { text, answer_start }
                      ]
                    }
                  ]
                }
              ]
            }
          ]
        }
        """
        file_data = _read_json(path + '/test', max_examples=self.max_examples)
        examples: List[Dict[str, Any]] = []
        # Some MLQA dumps include 'language' at top-level or file naming encodes language. Try to detect.
        detected_lang = lang
        for article in file_data:
            title = article.get("title", "")
            for paragraph in article.get("paragraphs", []):
                context = paragraph.get("context", "")
                for qa in paragraph.get("qas", []):
                    qid = qa.get("id") or qa.get("qid") or f"{title}_{len(examples)}"
                    question_text = qa.get("question") or qa.get("query") or ""
                    answers = []
                    for a in qa.get("answers", []):
                        txt = a.get("text") if isinstance(a, dict) else a
                        if txt:
                            answers.append(txt)
                    # Some MLQA golds have 'answers' absent -> unanswerable
                    examples.append({
                        "id": str(qid),
                        "title": title,
                        "question": question_text,
                        "answers": answers,
                        "context": context,
                        "lang": detected_lang,
                        "relevant_doc_ids": qa.get("relevant_doc_ids", []) or []
                    })
                    if self.max_examples and len(examples) >= self.max_examples:
                        return examples
        # print(examples[-1])
        return examples


    def _load_mkqa_jsonl(self, path: str, lang: str = None) -> List[Dict[str, Any]]:
        """
        MKQA is distributed as JSONL where each line is an example:
        {
          "query": "...",
          "answers": {"en":[{"text":"...","aliases":[...]}], "ru":[...], ...},
          "queries": {"en":"...","ru":"...", ...},
          "example_id": 12345
        }
        Creates one normalized example for the requested language (self.lang) if provided,
        else default to 'en' fallback.
        """
        examples: List[Dict[str, Any]] = []
        items = _read_jsonl(path + "/ext")
        requested_lang = lang or "en"
        for it in items:
            ex_id = it.get("example_id") or it.get("id") or None
            # prefer queries[field] if available
            q_map = it.get("queries") or {}
            ans_map = it.get("answers") or {}
            question = q_map.get(requested_lang) or it.get("query") or q_map.get("en") or ""
            # gather answers: ans_map might contain list of dicts for language
            answers_list = []
            if isinstance(ans_map, dict):
                entries = ans_map.get(requested_lang) or ans_map.get("en") or []
                for ent in entries:
                    if isinstance(ent, dict):
                        txt = ent.get("text")
                        if txt:
                            answers_list.append(txt)
                    elif isinstance(ent, str):
                        answers_list.append(ent)
            else:
                # not expected; try to fall back
                if isinstance(it.get("answers"), list):
                    for ent in it.get("answers", []):
                        if isinstance(ent, dict) and "text" in ent:
                            answers_list.append(ent["text"])
            examples.append({
                "id": str(ex_id) if ex_id is not None else f"mkqa_{len(examples)}",
                "question": question,
                "answers": answers_list,
                "context": "",   # MKQA typically doesn't include a context passage
                "lang": requested_lang
            })
            if self.max_examples and len(examples) >= self.max_examples:
                break
        return examples


    def _load_tydiqa(self, path: str, lang: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        TyDiQA loader.

        TyDiQA JSON structure varies by release, this method handles the below structure:
        {   
            "passage_answer_candidates":
                {
                    "plaintext_start_byte":[5,378,956,1342,3245,3801,3838,4063],
                    "plaintext_end_byte":[377,955,1317,3231,3792,3830,4057,4155]
                },
            "question_text":"Is Creole a pidgin of French?",
            "document_title":"French-based creole languages",
            "language":"english",
            "annotations":
                {
                    "passage_answer_candidate_index":[1],
                    "minimal_answers_start_byte":[-1],
                    "minimal_answers_end_byte":[-1],
                    "yes_no_answer":["YES"]
                },
            "document_plaintext":"\n\n\n\n\nPart of a series on theFrench language\nLangues d",
            "document_url":"https://en.wikipedia.org/wiki/French-based%20creole%20languages"
        }
        data -> paragraphs -> qas.
        """
        article_list = _read_jsonl(path)
        examples: List[Dict[str, Any]] = []

        for idx, art in enumerate(article_list):
            context = art.get("document_plaintext", "")
            question = art.get("question_text", "")
            qid = f"tydi_{idx}"

            answers = []
            ann = art.get("annotations", {})

            # -----------------------------
            # YES/NO answers
            # -----------------------------
            if isinstance(ann, dict):
                yn = ann.get("yes_no_answer", [])
                if yn and yn[0] not in ["NONE", None]:
                    answers.append(yn[0])

            # -----------------------------
            # Minimal answer byte spans
            # -----------------------------
            if not answers and isinstance(ann, dict):
                starts = ann.get("minimal_answers_start_byte", [])
                ends = ann.get("minimal_answers_end_byte", [])

                for s, e in zip(starts, ends):
                    if s >= 0 and e > s and e <= len(context):
                        answers.append(context[s:e])

            # -----------------------------
            # Passage candidate spans fallback
            # -----------------------------
            if not answers:
                cand = art.get("passage_answer_candidates", {})
                starts = cand.get("plaintext_start_byte", [])
                ends = cand.get("plaintext_end_byte", [])

                for s, e in zip(starts, ends):
                    if s >= 0 and e > s and e <= len(context):
                        answers.append(context[s:e])
                        break  # one is enough fallback

            detected_lang = (lang or art.get("language") or self._infer_lang_from_path(path))

            examples.append({
                "id": qid,
                "question": question,
                "answers": answers,
                "context": context,
                "lang": detected_lang,
                "relevant_doc_ids": []
            })

            if self.max_examples and len(examples) >= self.max_examples:
                break

        return examples


    def _load_squad(self, path: str, lang: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load SQuAD-like JSON (used by XQuAD).
        {
            "data": [ { 
                'title': ..., 
                'paragraphs': [ 
                    { 
                        'context': '...', 
                        'qas': [ { 'id':.., 'question':..., 'answers':[{'text':...}]} ] 
                    } 
                ] 
            } ]
        }
        """
        data_list = _read_json(path, max_examples=self.max_examples)
        # from pprint import pprint
        # # pprint(data_list[0]['paragraphs'], width=150, depth=10)
        # pprint(data_list[0], width=150, depth=10)
        # print(type(data_list), len(data_list[0]["paragraphs"]))
        # exit()
        examples: List[Dict[str, Any]] = []
        detected_lang = lang or self._infer_lang_from_path(path)
        for article in data_list:
            title = article.get("title", "")
            for para in article.get("paragraphs", []):
                context = para.get("context", "")
                for qa in para.get("qas", []):
                    qid = qa.get("id") or qa.get("qid") or f"s(x)quad_{len(examples)}"
                    question_text = qa.get("question") or ""
                    answers = []
                    for a in qa.get("answers", []):
                        if isinstance(a, dict) and "text" in a:
                            answers.append(a["text"])
                        elif isinstance(a, str):
                            answers.append(a)
                    examples.append({
                        "id": str(qid),
                        "title": title,
                        "question": question_text,
                        "answers": answers,
                        "context": context,
                        "lang": detected_lang,
                        "relevant_doc_ids": []
                    })
                    if self.max_examples and len(examples) >= self.max_examples:
                        return examples
        return examples


    def _load_corpus(self, path: str, max_examples: Optional[int] = None, lang: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load corpus (wiki/ccnews) as documents (no QA). Returns entries with empty 'question' and 'answers'.
        If 'path' is a directory, load all files inside.
        """
        docs = []
        if os.path.isdir(path):
            docs = _read_json(path, max_examples)
            examples = []
            for i in range(len(docs)):  # article in docs:
                article = docs[i]
                examples.append({
                    "id": article["id"] if article.get("id") is not None else article.get("_global_idx", ""),
                    'publish_date': article.get("date_publish", ""),
                    "title": article.get("title", ""),
                    "text": article.get("text", ""),
                    "lang": lang or self.lang,
                    "url": article.get("url", ""),
                })

                if len(examples) >= max_examples:
                    return examples

        else:
            warnings.warn(f"Corpus path {path} does not exist.")
        if max_examples:
            return examples[:max_examples]
        return examples


    # ---------------------------
    # Util.
    # ---------------------------
    def _infer_lang_from_path(self, path: str) -> str:
        """
        Try to infer language code from filename (e.g., dev-context-ar-question-ar.json -> 'ar')
        """
        base = os.path.basename(path).lower()
        # common iso codes
        for code in ("en", "hi", "ru", "de", "es"):
            if f".{code}." in base or f"_{code}." in base or f"-{code}." in base or f"_{code}" in base:
                return code
            if base.endswith(f"_{code}.json") or base.endswith(f"-{code}.json") or base.endswith(f".{code}.json"):
                return code
        # default
        return self.lang or "en"




"""         TESTING CODE        """


# import time
# QA_PATH = "./data/qa"

# # ✅
# print(f"\n{'*'*50} QA Dataset {'*'*50}")
# for qa in ['mkqa', 'tydiqa', 'mlqa', 'xquad']:
#     start_t = time.time()
#     dataset_loader = DatasetLoader(qa, file_path=f"{QA_PATH}/{qa}", max_examples=100000)
#     dataset = dataset_loader.load()

#     end_t = time.time()
#     print(f"\nTime taken to load {qa.upper()} dataset: {end_t-start_t:.2f} sec")
#     print(f"length: {len(dataset)}")

# # ✅
# print(f"\n\n{'*'*50} Corpus Dataset {'*'*50}")
# for ind in ['ccnews', 'wiki']:
#     for lang in ["es", "hi", "ru", "en", "de"]:
#         print(f"\n{'=='*50}")
#         start_t = time.time()

#         dataset_loader = DatasetLoader(ind, file_path=f"./data/index/hf_{ind}_extracted/{lang}", lang=lang, max_examples=100000)
#         dataset = dataset_loader.load()

#         # print(len(dataset))
#         end_t = time.time()
#         print(f"Time taken to load {lang.upper()} data {ind.upper()} dataset: {end_t-start_t:.2f} sec")



# # OUTPUT

# # """
# # (__venv) PS D:\XRAG-plus> python -m src.evaluation.dataset_loader

# # ************************************************** QA Dataset **************************************************

# # Time taken to load MKQA dataset: 1.10 sec
# # length: 10000

# # Time taken to load TYDIQA dataset: 10.63 sec
# # length: 22014

# # Time taken to load MLQA dataset: 6.05 sec
# # length: 100000

# # Time taken to load XQUAD dataset: 2.34 sec
# # length: 100000


# # ************************************************** Corpus Dataset **************************************************

# # ====================================================================================================
# # Total data length : 100000
# # Time taken to load ES data CCNEWS dataset: 6.72 sec

# # ====================================================================================================
# # Total data length : 100000
# # Time taken to load HI data CCNEWS dataset: 6.97 sec

# # ====================================================================================================
# # Total data length : 100000
# # Time taken to load RU data CCNEWS dataset: 8.15 sec

# # ====================================================================================================
# # Total data length : 100000
# # Time taken to load EN data CCNEWS dataset: 6.24 sec

# # ====================================================================================================
# # Total data length : 100000
# # Time taken to load DE data CCNEWS dataset: 6.33 sec

# # ====================================================================================================
# # Total data length : 100000
# # Time taken to load ES data WIKI dataset: 20.03 sec

# # ====================================================================================================
# # Total data length : 100000
# # Time taken to load HI data WIKI dataset: 9.10 sec

# # ====================================================================================================
# # Total data length : 100000
# # Time taken to load RU data WIKI dataset: 25.05 sec

# # ====================================================================================================
# # Total data length : 100000
# # Time taken to load EN data WIKI dataset: 9.93 sec

# # ====================================================================================================
# # Total data length : 100000
# # Time taken to load DE data WIKI dataset: 24.10 sec
# # """