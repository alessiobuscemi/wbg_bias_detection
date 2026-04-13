import json
import os


class ConfigLoader:
    """
    Loads exercise configuration from JSON files.

    Separating config from code means facilitators can update
    templates and community values without opening the notebook.
    """

    def __init__(self, templates_path: str, communities_path: str):
        self._templates_path   = templates_path
        self._communities_path = communities_path

    # ── Public ───────────────────────────────────────────────

    def load(self) -> tuple[dict, dict]:
        """
        Load and return (templates, communities).
        Raises FileNotFoundError with a clear message if either
        file is missing.
        """
        templates   = self._load_json(self._templates_path,   "templates")
        communities = self._load_json(self._communities_path, "communities")
        return templates, communities

    # ── Private ───────────────────────────────────────────────

    def _load_json(self, path: str, label: str) -> dict:
        """Read a JSON file and return its contents as a dict."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Could not find {label} file at '{path}'. "
                f"Check that the repository was cloned correctly."
            )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def summarise(templates: dict, communities: dict) -> None:
        """Print a brief overview of the loaded configuration."""
        print("Configuration loaded.")
        print(f"  Templates   : {list(templates.keys())}")
        for group, values in communities.items():
            print(f"  [{group}] : {values}")
