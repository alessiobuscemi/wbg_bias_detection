import csv
import json
import os
from collections import defaultdict

from models import Prompt


class PromptExporter:
    """Serialises a list of Prompt objects to CSV and JSON."""

    def __init__(self, prompts: list[Prompt]):
        self._prompts = prompts

    # ── Public ───────────────────────────────────────────────

    def export_all(self, csv_path: str, json_path: str) -> None:
        """Write both export files and download the CSV in Colab."""
        self._to_csv(csv_path)
        self._to_response_template(json_path)
        self._download([csv_path])

    # ── Private ───────────────────────────────────────────────

    def _to_csv(self, path: str) -> None:
        """Write a print-friendly CSV with one row per prompt."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["prompt_id", "template_id", "values", "prompt"]
            )
            writer.writeheader()
            for p in self._prompts:
                writer.writerow({
                    "prompt_id":   p.prompt_id,
                    "template_id": p.template_id,
                    "values":      "; ".join(f"{k}={v}" for k, v in p.values.items()),
                    "prompt":      p.prompt,
                })
        print(f"  ✅ Saved: {path}")

    def _to_response_template(self, path: str) -> None:
        """
        Write a grouped JSON file merging with any existing file.

        Merge rules:
          - If a template group already exists in the file, it is
            kept exactly as-is — entries, responses, and all.
            New communities added to COMMUNITIES do not affect it.
          - If a template group is new (its template_id is not in
            the existing file), it is appended with all current
            community expansions and empty response fields.
          - Template groups that have been removed from TEMPLATES
            are dropped from the file.
        """
        existing_by_tid = self._load_existing(path)
        new_by_tid      = self._group_by_template()
        merged          = self._merge(existing_by_tid, new_by_tid)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

        added   = [tid for tid in new_by_tid if tid not in existing_by_tid]
        kept    = [tid for tid in new_by_tid if tid in existing_by_tid]
        dropped = [tid for tid in existing_by_tid if tid not in new_by_tid]

        print(f"  ✅ Saved: {path}")
        if kept:
            print(f"     Preserved (unchanged) : {kept}")
        if added:
            print(f"     Added (new templates) : {added}")
        if dropped:
            print(f"     Dropped (removed)     : {dropped}")

    def _load_existing(self, path: str) -> dict[str, dict]:
        """
        Load the existing response file if present and index its
        groups by template_id. Returns an empty dict if no file
        exists yet or if the file is malformed.
        """
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                groups = json.load(f)
            return {g["template_id"]: g for g in groups}
        except json.JSONDecodeError as e:
            print(f"  ⚠️  Existing file is malformed ({e}) — starting fresh.")
            print(f"     A backup has been saved to {path}.bak")
            import shutil
            shutil.copy(path, f"{path}.bak")
            return {}

    def _group_by_template(self) -> dict[str, list[Prompt]]:
        """Group the current prompt list by template_id."""
        groups = defaultdict(list)
        for p in self._prompts:
            groups[p.template_id].append(p)
        return dict(groups)

    def _merge(
        self,
        existing_by_tid: dict[str, dict],
        new_by_tid: dict[str, list[Prompt]],
    ) -> list[dict]:
        """
        Produce the merged group list.

        For each template in new_by_tid:
          - If it already exists in the file, use the existing group
            verbatim (responses intact, communities frozen).
          - If it is new, build a fresh group with blank responses.
        """
        merged = []
        for tid, prompts in new_by_tid.items():
            if tid in existing_by_tid:
                merged.append(existing_by_tid[tid])
            else:
                merged.append({
                    "template_id":        tid,
                    "reference_template": prompts[0].reference_template,
                    "entries": [
                        {
                            "prompt_id": p.prompt_id,
                            "values":    p.values,
                            "prompt":    p.prompt,
                            "response":  "",
                        }
                        for p in prompts
                    ],
                })
        return merged

    @staticmethod
    def _download(paths: list[str]) -> None:
        """Download files to the local machine (Colab only)."""
        try:
            from google.colab import files
            for path in paths:
                files.download(path)
            print(f"  📥 Downloaded: {', '.join(paths)}")
        except ImportError:
            print("  (Not in Colab — files saved in the current directory.)")
