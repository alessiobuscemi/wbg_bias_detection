from signal_classifier import SignalClassifier
from llm_judge import LLMJudge


class CascadeClassifier:
    """
    Combines SignalClassifier and LLMJudge into a two-stage pipeline.

    Stage 1: heuristic (free, instant).
    Stage 2: LLM judge (API call) — only invoked when Stage 1
             returns UNCLEAR.
    """

    def __init__(self, judge: LLMJudge):
        self._heuristic = SignalClassifier()
        self._judge     = judge

    def classify(self, text: str) -> tuple[str, str]:
        """
        Classify a response text.

        Returns a tuple (signal, method) where method is either
        'heuristic' or 'llm_judge', indicating which stage
        produced the final signal.
        """
        signal = self._heuristic.classify(text)
        if signal != "UNCLEAR":
            return signal, "heuristic"

        print("    → Heuristic returned UNCLEAR, calling LLM judge…")
        signal = self._judge.classify(text)
        return signal, "llm_judge"
