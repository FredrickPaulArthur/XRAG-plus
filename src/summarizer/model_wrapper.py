"""
Wrappers around ML model backends.

- HFWrapper: HuggingFace pipeline-based summarizer (lazy import)
- OpenAIWrapper: calls OpenAI if configured (lazy import and safe fallback)

These wrappers purposely avoid hard-dependencies at import time; they raise helpful
errors instructing the user how to install optional extras.
"""
from __future__ import annotations




class HFWrapper:
    def __init__(self, model: str, temperature: float, device: str= "cpu"):
        self.model = model
        self._pipeline = None
        self.device = device
        self.temperature = temperature

    def _init_pipeline(self):
        try:
            from transformers import pipeline
        except ImportError as e:
            raise RuntimeError("HFWrapper requires `transformers`. Install with: pip install transformers") from e
        try:
            self._pipeline = pipeline(task="summarization", model=self.model, device=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HF summarization pipeline model={self.model}, device={self.device}.") from e

    def summarize(self, texts, **kwargs):
        if self._pipeline is None:
            self._init_pipeline()
        return self._pipeline(texts, **kwargs)      # pipeline supports lists, for multiple Docs.



class OpenAIWrapper:
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def summarize(self, texts, **kwargs):
        # Lazy import and graceful error
        try:
            import openai
        except Exception as exc:
            raise RuntimeError("openai package required for OpenAIWrapper. Install with `pip install openai`") from exc
        if self.api_key:
            openai.api_key = self.api_key
        # For safety, accept either single str or list
        prompts = texts if isinstance(texts, (list, tuple)) else [texts]
        out = []
        for p in prompts:
            prompt = ("Summarize the following text in clear, concise English. Keep it factual and to the point.\n\n" + p)
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=kwargs.get("max_tokens", 256),
            )
            # Best-effort extraction
            summary = resp["choices"][0]["message"]["content"].strip()
            out.append({"summary_text": summary})
        return out




class CohereAIWrapper:
    """
    Wrapper for Cohere summarization. Tries to use the dedicated `summarize`
    endpoint if available, otherwise falls back to generation/chat endpoints.

    Example usage:
        wrapper = CohereAIWrapper(api_key="...", model="command-r", temperature=0.0)
        out = wrapper.summarize("Long article text here")
        out -> [{"summary_text": "..."}]
    """

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self.api_key = api_key
        self.model = model
        self.temperature = float(temperature or 0.0)
        self.max_tokens = max_tokens
        self.length = "auto"        # cohere specific
        self.format = "paragraph"   # cohere specific
        self.client = None

    def _init_client(self):
        try:
            import cohere
        except Exception as e:
            raise RuntimeError("cohere package required for CohereAIWrapper. Install with `pip install cohere`") from e

        # Uses the newer ClientV2, only to use the older client when there already was legacy code.
        if hasattr(cohere, "ClientV2"):
            self.client = cohere.ClientV2(self.api_key)
        else:
            raise RuntimeError("Installed cohere SDK does not expose a Client class V2.")

    def _extract_summary_text(self, resp) -> str:
        """
        Utility that tries multiple known shapes of Cohere responses and extracts a reasonable summary string.
        """
        if resp is None:
            return ""

        # 1) Named attribute 'summary' (object or list)
        if hasattr(resp, "summary"):
            s = getattr(resp, "summary")
            if isinstance(s, (list, tuple)) and s:
                s = s[0]
            # sometimes elements have .text
            if hasattr(s, "text"):
                return str(s.text).strip()
            return str(s).strip()

        # 2) dict-like responses
        try:
            # convert mapping-like objects to dict safely
            if isinstance(resp, dict):
                for k in ("summary", "summary_text", "summaries", "prediction", "output"):
                    if k in resp:
                        val = resp[k]
                        if isinstance(val, (list, tuple)) and val:
                            val = val[0]
                        if isinstance(val, dict) and "text" in val:
                            return str(val["text"]).strip()
                        return str(val).strip()
        except Exception:
            pass

        # 3) 'generations' attribute used by older cohere.generate responses
        if hasattr(resp, "generations"):
            gens = getattr(resp, "generations")
            if isinstance(gens, (list, tuple)) and gens:
                g0 = gens[0]
                if hasattr(g0, "text"):
                    return str(g0.text).strip()
                # sometimes nested
                if isinstance(g0, dict) and "text" in g0:
                    return str(g0["text"]).strip()

        # 4) simple text attribute
        if hasattr(resp, "text"):
            return str(getattr(resp, "text")).strip()

        # 5) final fallback to string representation
        return str(resp).strip()

    def summarize(self, texts):
        """
        Summarize a single string or a list of strings.

        Args:
            texts: str or List[str]
            length: 'short' | 'medium' | 'long' | 'auto' (Cohere's summarize length param)
            format: 'paragraph' | 'bullets' etc. (Cohere accepts format param)

        Returns:
            List[Dict[str, str]] where each dict has key 'summary_text'
            (if input was a single str, a list with one element is still returned,
             to match other wrappers' behavior).
        """
        if self.client is None:
            self._init_client()

        items = texts if isinstance(texts, (list, tuple)) else [texts]      # Normalize to list for uniform processing
        results = []

        for text in items:
            text = "" if text is None else str(text)
            if not text:
                results.append({"summary_text": ""})
                continue
            try:
                # ClientV2 chat-like interface may be named 'chat' or similar
                if hasattr(self.client, "chat"):
                    from src.summarizer.prompts import summ_prompt
                    messages = [{"role": "user", "content": summ_prompt + text}]

                    resp = self.client.chat(model=self.model, messages=messages, temperature=self.temperature, max_tokens=90)
                    # print(resp.message.content[0].text)

                    # summary_text = self._extract_summary_text(resp)
                    summary_text = resp.message.content[0].text
                    results.append({"summary_text": summary_text})
                    continue
            except Exception as e:
                print(f"\nCohereAI Wrapper Chat call has failed: {e}")

            # Last-resort: return empty string or the truncated original if nothing worked
            results.append({"summary_text": text[:512]})

        return results



"""
Original: 

The Soviet Union (USSR) emerged in 1922 after years of civil war and political upheaval following the Russian Revolution. 
Built on a single-party communist system, it rapidly industrialized under central planning and became one of the world's 
leading powers. The USSR played a decisive role in the defeat of Nazi Germany during World War II and later entered a prolonged 
ideological conflict with the United States known as the Cold War. Despite achievements in science, heavy industry, 
and education, the Soviet system struggled with economic inefficiencies, shortages, and political repression. 
By the late 1980s, reforms under Mikhail Gorbachev—glasnost and perestroika—exposed deep structural issues. 
In 1991, rising nationalist movements and economic crises culminated in the dissolution of the USSR into fifteen independent 
republics.


Summaries: (max_tokens=90, temperature=0)

The Soviet Union, established in 1922, rose to global prominence but faced internal challenges. Under Gorbachev's reforms, 
the USSR's economic and political weaknesses were revealed, leading to its dissolution in 1991, resulting in the formation 
of multiple independent republics.

The Soviet Union, established in 1922, rose to global prominence as a communist superpower, significantly impacting 
World War II's outcome. However, economic challenges and political reforms under Gorbachev led to its dissolution in 1991, 
resulting in the formation of fifteen new republics. This period marked the end of the Cold War era.

The Soviet Union, established in 1922, rose to global prominence but faced internal challenges. Under Gorbachev's reforms, 
the late 1980s revealed economic and political problems, leading to the USSR's dissolution in 1991 into multiple republics. 
This marked the end of a significant era in world history.

The Soviet Union, established in 1922, rose to global prominence but faced challenges. Under Mikhail Gorbachev's leadership, 
attempts at reform in the late 1980s revealed systemic problems, leading to its dissolution in 1991 into multiple republics 
due to economic crises and nationalist sentiments
"""