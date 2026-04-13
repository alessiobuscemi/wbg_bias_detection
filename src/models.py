from dataclasses import dataclass, field


@dataclass
class Prompt:
    """
    A single fully-expanded prompt ready for chatbot testing.

    prompt_id          Unique sequential ID, e.g. "P001"
    template_id        Source template key, e.g. "T3"
    reference_template Original template string with placeholders
    values             Mapping of placeholder → substituted value
    prompt             Fully expanded prompt text (with answer constraint appended)
    """
    prompt_id: str
    template_id: str
    reference_template: str
    values: dict
    prompt: str


@dataclass
class Response:
    """
    A Prompt paired with the raw chatbot answer collected during
    manual testing.

    response    The raw chatbot response text (empty until filled in)
    run_name    Optional label to distinguish multiple testing runs
                (e.g. "run1", "gpt4o") — defaults to empty string
    """
    prompt_id: str
    template_id: str
    reference_template: str
    values: dict
    prompt: str
    response: str
    run_name: str = ""


@dataclass
class TemplateResult:
    """
    Analysis outcome for one template group (all community variants
    of a single template evaluated together, across all runs).

    responses          All Response objects in this group
    signals            Classified signal per response (YES/NO/CONDITIONAL/UNCLEAR)
    lengths            Character count per response
    differential       True if signals diverge across communities
    length_std         Standard deviation of response lengths —
                       high values suggest unequal engagement depth
    length_std_warning True when length_std exceeds the threshold
    """
    template_id: str
    reference_template: str
    responses: list[Response]
    signals: list[str]
    lengths: list[int]
    differential: bool
    length_std: float | None
    length_std_warning: bool

    @property
    def verdict(self) -> str:
        """PASS if all signals agree, FAIL if they diverge."""
        return "FAIL" if self.differential else "PASS"
