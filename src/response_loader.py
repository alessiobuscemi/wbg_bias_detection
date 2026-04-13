import json
import os

from models import Response


class ResponseLoader:
    """
    Loads a completed response JSON file and deserialises each
    entry into a typed Response object.

    Tries to read from disk first so that users who wrote the
    file inline or mounted it from Drive do not need to go
    through the upload dialog.
    """

    def __init__(self, path: str):
        self._path = path

    # ── Public ───────────────────────────────────────────────

    def load(self) -> list[Response]:
        """Load and return a flat list of Response objects."""
        raw = self._read_raw()
        return [Response(**entry) for entry in self._flatten(raw)]

    # ── Private ───────────────────────────────────────────────

    def _read_raw(self) -> list[dict]:
        """
        Read raw JSON from disk if available, otherwise open the
        Colab upload dialog.
        """
        if os.path.exists(self._path):
            print(f"📂 Found '{self._path}' on disk — loading directly.")
            return self._read_from_disk(self._path)
        print(f"📂 '{self._path}' not found — opening upload dialog.")
        return self._read_from_colab()

    @staticmethod
    def _read_from_disk(path: str) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _read_from_colab() -> list[dict]:
        try:
            from google.colab import files
            uploaded = files.upload()
            filename = list(uploaded.keys())[0]
            return json.loads(uploaded[filename].decode("utf-8"))
        except ImportError:
            raise FileNotFoundError(
                "File not found and not running in Colab — "
                "place the response file in the expected path."
            )

    @staticmethod
    def _flatten(raw: list[dict]) -> list[dict]:
        """
        Flatten the grouped JSON format into a flat list of dicts,
        re-attaching template_id and reference_template to each
        entry before deserialisation.
        """
        flat = []
        for group in raw:
            for entry in group["entries"]:
                flat.append({
                    "template_id":        group["template_id"],
                    "reference_template": group["reference_template"],
                    **entry,
                })
        return flat

    @staticmethod
    def summarise(responses: list[Response]) -> None:
        """Print a brief overview of the loaded dataset."""
        filled  = sum(1 for r in responses if r.response.strip())
        missing = len(responses) - filled
        print(f"\n  Entries loaded      : {len(responses)}")
        print(f"  Responses filled in : {filled}")
        if missing:
            print(f"  ⚠️  Missing          : {missing}  (skipped in analysis)")
