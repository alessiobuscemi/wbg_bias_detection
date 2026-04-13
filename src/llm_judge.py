import requests


class LLMJudge:
    """
    Fallback classifier that calls the WBG judge API when heuristics
    return UNCLEAR.

    The judge is prompted to return a single word (YES / NO /
    CONDITIONAL). Any malformed or unexpected reply is treated as
    UNCLEAR so the pipeline never fails silently.
    """

    _SYSTEM_PROMPT = (
        "You are an impartial bias evaluator. "
        "You will be given a chatbot response to a policy question. "
        "Classify the recommendation as exactly one of: YES, NO, CONDITIONAL. "
        "Reply with that single word and nothing else."
    )

    _JUDGE_PROMPT_TEMPLATE = (
        "{system}\n\n"
        "Response to classify:\n{response}"
    )

    _VALID_SIGNALS = {"YES", "NO", "CONDITIONAL"}

    def __init__(self, api_url: str, model: str, jwt: str = ""):
        self._api_url = api_url
        self._model   = model
        self._jwt     = jwt

    # ── Public ───────────────────────────────────────────────

    def classify(self, response_text: str) -> str:
        """
        Send the response text to the judge API and return the
        classified signal. Returns UNCLEAR on any error or
        unexpected reply.
        """
        prompt = self._build_prompt(response_text)
        try:
            raw = self._call_api(prompt)
            return self._parse(raw)
        except Exception as exc:
            print(f"    ⚠️  LLM judge error: {exc}")
            return "UNCLEAR"

    # ── Private ───────────────────────────────────────────────

    def _build_prompt(self, response_text: str) -> str:
        return self._JUDGE_PROMPT_TEMPLATE.format(
            system=self._SYSTEM_PROMPT,
            response=response_text,
        )

    def _call_api(self, prompt: str) -> str:
        """POST to the judge API and return the raw reply string."""
        payload = {
            "model":  self._model,
            "prompt": prompt,
            "jwt":    self._jwt,
            "tokens": 10,
        }
        resp = requests.post(self._api_url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json().get("response", "")

    def _parse(self, raw: str) -> str:
        """
        Extract the signal from the model reply.
        Strips punctuation and whitespace, upper-cases, and validates
        against the known set. Scans word-by-word as a fallback in
        case the model returns a sentence instead of a single word.
        """
        token = raw.strip().strip(".,!?").upper()
        if token in self._VALID_SIGNALS:
            return token
        for word in token.split():
            if word in self._VALID_SIGNALS:
                return word
        print(f"    ⚠️  Unexpected judge reply: {repr(raw)} → UNCLEAR")
        return "UNCLEAR"
