"""Export a ReviewReport to plain-text formats (TXT, MD, SRT, VTT)."""

from __future__ import annotations

from approach_2.src.models import AlignedSegment, ReviewReport

_TIER_LABEL = {
    "auto_accept": "auto-accept",
    "review_technical": "review-if-technical",
    "mandatory": "mandatory",
}


def _best_text(s: AlignedSegment) -> str:
    if s.engine_a is not None:
        return s.engine_a.text
    if s.engine_b is not None:
        return s.engine_b.text
    return ""


def to_txt(report: ReviewReport) -> str:
    lines = [f"Audio: {report.audio}", f"Engines: {', '.join(report.engines)}", ""]
    for s in report.segments:
        lines.append(
            f"[{s.idx:02d}] {s.start:7.2f}-{s.end:7.2f}s  conf={s.confidence:3.0f}  {s.tier}"
        )
        lines.append(f"      {_best_text(s)}")
    return "\n".join(lines) + "\n"


def to_md(report: ReviewReport) -> str:
    lines = [f"# Review report: {report.audio}", "", f"Engines: {', '.join(report.engines)}", ""]
    lines.append("| # | span | conf | tier | text |")
    lines.append("|---|---|---|---|---|")
    for s in report.segments:
        text = _best_text(s).replace("|", "\\|")
        lines.append(
            f"| {s.idx} | {s.start:.2f}-{s.end:.2f}s | {s.confidence} | {s.tier} | {text[:80]} |"
        )
    return "\n".join(lines) + "\n"


def _srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_time(seconds: float) -> str:
    return _srt_time(seconds).replace(",", ".")


def to_srt(report: ReviewReport) -> str:
    blocks = []
    for n, s in enumerate(report.segments, 1):
        blocks.append(
            f"{n}\n{_srt_time(s.start)} --> {_srt_time(s.end)}\n{_best_text(s)}\n"
        )
    return "\n".join(blocks)


def to_vtt(report: ReviewReport) -> str:
    blocks = ["WEBVTT\n"]
    for s in report.segments:
        blocks.append(f"{_vtt_time(s.start)} --> {_vtt_time(s.end)}\n{_best_text(s)}\n")
    return "\n".join(blocks)
