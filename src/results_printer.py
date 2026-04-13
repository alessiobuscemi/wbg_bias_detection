from models import TemplateResult


class ResultsPrinter:
    """
    Formats and prints a TemplateResult report to stdout.
    Produces a per-template detail section and a summary table.
    """

    _VERDICT_ICON = {"PASS": "✅", "FAIL": "❌"}
    _WARN_ICON    = "⚠️ "
    _WIDTH        = 68

    def __init__(self, results: dict[str, TemplateResult]):
        self._results = results

    # ── Public ───────────────────────────────────────────────

    def print_report(self) -> None:
        """Print the full report to stdout."""
        self._print_header("BIAS EVALUATION REPORT")
        for res in self._results.values():
            self._print_template_block(res)
        self._print_summary()

    # ── Private — block printers ──────────────────────────────

    def _print_header(self, title: str) -> None:
        print("=" * self._WIDTH)
        print(f"  {title}")
        print("=" * self._WIDTH)

    def _print_template_block(self, res: TemplateResult) -> None:
        icon = self._VERDICT_ICON[res.verdict]
        print(f"\n{'─'*self._WIDTH}")
        print(f"  Template {res.template_id}  —  {icon}  {res.verdict}")
        print(f"  {res.reference_template}")
        print(f"{'─'*self._WIDTH}")

        for entry, sig, length in zip(res.responses, res.signals, res.lengths):
            vals_str = "  |  ".join(f"{k} = {v}" for k, v in entry.values.items())
            print(f"  [{entry.prompt_id}]  {vals_str}")
            print(f"           Signal : {sig:<20}  Length : {length} chars")
            preview = entry.response[:220].replace("\n", " ")
            if len(entry.response) > 220:
                preview += "…"
            print(f"           {preview}\n")

        self._print_verdict_line(res)

    def _print_verdict_line(self, res: TemplateResult) -> None:
        if res.differential:
            unique = sorted({s.split()[0] for s in res.signals if not s.startswith("UNCLEAR")})
            print(f"  {self._VERDICT_ICON['FAIL']} Differential treatment detected: {unique}")
        else:
            print(f"  {self._VERDICT_ICON['PASS']} Consistent signals: {res.signals}")

        if res.length_std is not None:
            warn = self._WARN_ICON if res.length_std_warning else "   "
            print(
                f"  {warn} Response length std dev : {res.length_std:.0f} chars"
                + (" — unequal engagement depth suspected" if res.length_std_warning else "")
            )

    def _print_summary(self) -> None:
        total_fail = sum(1 for r in self._results.values() if r.differential)
        total_warn = sum(1 for r in self._results.values() if r.length_std_warning)

        print(f"\n{'='*self._WIDTH}")
        print("  SUMMARY")
        print(f"{'─'*self._WIDTH}")
        print(f"  {'Template':<12} {'Verdict':<8} {'Std dev':>10}  {'Signals'}")
        print(f"  {'─'*10:<12} {'─'*6:<8} {'─'*8:>10}  {'─'*32}")
        for tid, res in self._results.items():
            std_str = f"{res.length_std:.0f} chars" if res.length_std is not None else "—"
            warn    = f" {self._WARN_ICON}" if res.length_std_warning else ""
            print(f"  {tid:<12} {res.verdict:<8} {std_str:>10}{warn}  {res.signals}")
        print(f"{'─'*self._WIDTH}")
        print(f"  Templates evaluated       : {len(self._results)}")
        print(f"  Tests failed              : {total_fail}  (differential signals)")
        print(f"  Length warnings           : {total_warn}  (std dev ≥ 100 chars)")
        print(f"{'='*self._WIDTH}")
