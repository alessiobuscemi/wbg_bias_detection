import re
import itertools

from models import Prompt


class PromptExpander:
    """
    Expands templates into all combinations of community values.

    Each {PLACEHOLDER} found in a template is replaced with every
    value defined for that group in the communities dict. Templates
    with multiple placeholders produce the full Cartesian product
    of all value combinations.

    Every expanded prompt has a constraint sentence appended that
    instructs the model to begin its answer with Yes / No / It depends
    before providing its reasoning.
    """

    _PLACEHOLDER_RE = re.compile(r'\{([A-Z_]+)\}')

    # Appended to every prompt to constrain the response format.
    # Keeping it here makes it easy to adjust wording in one place.
    _ANSWER_CONSTRAINT = (
        "Please start your answer with one of the following: "
        "Yes, No, or It depends — then provide your reasoning."
    )

    def __init__(self, templates: dict, communities: dict):
        self._templates   = templates
        self._communities = communities

    # ── Public ───────────────────────────────────────────────

    def expand_all(self) -> list[Prompt]:
        """
        Expand every template and return the full list of Prompt
        objects with sequential IDs assigned.
        """
        prompts = []
        for tid, text in self._templates.items():
            prompts.extend(self._expand_one(tid, text))

        for i, p in enumerate(prompts, 1):
            p.prompt_id = f"P{i:03d}"

        return prompts

    # ── Private ───────────────────────────────────────────────

    def _expand_one(self, template_id: str, text: str) -> list[Prompt]:
        """
        Expand a single template. Returns an empty list and prints a
        warning if a placeholder has no matching community group.
        """
        placeholders = self._PLACEHOLDER_RE.findall(text)

        if not placeholders:
            return [self._make_prompt(template_id, text, text, {})]

        ph_values = []
        for ph in placeholders:
            if ph not in self._communities:
                print(f"  ⚠️  {{{ph}}} in {template_id} has no matching community group — skipping.")
                return []
            ph_values.append((ph, self._communities[ph]))

        prompts = []
        for combo in itertools.product(*[vals for _, vals in ph_values]):
            filled  = text
            mapping = {}
            for (ph, _), val in zip(ph_values, combo):
                filled      = filled.replace(f"{{{ph}}}", val)
                mapping[ph] = val
            prompts.append(self._make_prompt(template_id, text, filled, mapping))
        return prompts

    def _make_prompt(self, template_id: str, reference: str, filled: str, values: dict) -> Prompt:
        """
        Construct a Prompt object with the answer constraint appended.
        The reference_template stores the original text without the
        constraint so it remains clean in the output file.
        ID is assigned later in expand_all.
        """
        constrained = f"{filled} {self._ANSWER_CONSTRAINT}"
        return Prompt(
            prompt_id="",
            template_id=template_id,
            reference_template=reference,
            values=values,
            prompt=constrained,
        )
