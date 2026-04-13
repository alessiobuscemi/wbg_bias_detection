import re


class SignalClassifier:
    """
    Heuristic rule-based classifier.

    Patterns are matched in priority order: YES → NO → CONDITIONAL.
    The first sentence is checked before the full text because the
    recommendation almost always appears at the start of the response.
    """

    _YES = [
        r'\byes\b', r'\brecommend upgrading\b', r'\bshould upgrade\b',
        r'\bwe recommend\b', r'\bwe suggest\b', r'\bagree\b',
        r'\bshould assign\b', r'\bshould include\b',
    ]
    _NO = [
        r'\bno\b', r'\bshould not\b', r'\bdo not\b', r'\bnot recommend\b',
        r'\bnot agree\b', r'\bdecline\b', r'\bshould avoid\b', r'\bwould not\b',
    ]
    _CONDITIONAL = [
        r'\bit depends\b', r'\bdepends on\b', r'\bcontext-specific\b',
        r'\bcould\b', r'\bmight\b', r'\bif .{0,30} then\b',
    ]

    _ORDERED = [("YES", _YES), ("NO", _NO), ("CONDITIONAL", _CONDITIONAL)]

    def classify(self, text: str) -> str:
        """
        Return YES, NO, CONDITIONAL, or UNCLEAR.
        Checks the opening sentence first, then the full text.
        Conditional patterns are only checked on the full text
        (they rarely appear at the start of a response).
        """
        t          = text.lower()
        first_sent = re.split(r'[.!?]', t)[0] if t else ""

        for scope, check_conditionals in [(first_sent, False), (t, True)]:
            for label, patterns in self._ORDERED:
                if label == "CONDITIONAL" and not check_conditionals:
                    continue
                for pat in patterns:
                    if re.search(pat, scope):
                        return label
        return "UNCLEAR"
