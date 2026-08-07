"""Layer 3: real-LLM evaluation of the shared scenario catalog.

This is the ONLY place that asks "is the LLM judge itself accurate?", and it is
deliberately kept out of the fast, offline test suite. It requires live API
keys and (for Approach 2) synthesizes a spoken audio clip of the gold text.

Run:
    python -m testing.eval_layer3            # both approaches
    python -m testing.eval_layer3 --approach 1   # approach 1 only

It reports aggregate category-accuracy over the catalog. It does NOT assert
exact equality with the catalog -- the judge is evaluated, not mocked. Exits
with code 0 iff aggregate accuracy meets the floor.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from testing import SCENARIOS

ACCURACY_FLOOR = 0.6  # category-match rate required to pass Layer 3

# Scenarios the real judge is expected to classify deterministically. semantic_only
# scenarios are excluded: their meaning is context-dependent and the judge may
# legitimately call them accurate.
_STRICT = [
    s for s in SCENARIOS if s.expected_a2_category is not None or s.expected_a1_category is not None
]


def _synth_gold_audio(text: str, out_dir: Path, voice: str = "en-US-JennyNeural") -> Path:
    """Synthesize a 16 kHz mono WAV of `text` (gold spoken content) with edge-tts."""
    mp3 = out_dir / "tts.mp3"
    wav = out_dir / "tts.wav"
    try:
        import asyncio
        import edge_tts

        async def _go():
            await edge_tts.Communicate(text, voice).save(str(mp3))

        asyncio.run(_go())
    except Exception as exc:  # offline / blocked network
        raise RuntimeError(f"edge-tts failed (needs network): {exc}") from exc
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(mp3), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(wav),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-300:]}")
    return wav


def evaluate_approach2() -> Counter:
    """Run the real Gemini judge over each strict scenario's gold audio."""
    from approach_2 import config
    if not config.GEMINI_API_KEY:
        print("  (skipped: GEMINI_API_KEY not set)")
        return Counter()

    from approach_2.src.judge import GeminiJudge, JudgeRequest
    from approach_2.src.models import AlignedSegment
    from approach_2.tests.fixtures import seg

    judge = GeminiJudge()
    results: Counter = Counter()
    with tempfile.TemporaryDirectory(prefix="layer3_a2_") as work:
        work = Path(work)
        for scenario in _STRICT:
            try:
                wav = _synth_gold_audio(scenario.baseline, work)
            except RuntimeError as exc:
                print(f"  [A2] {scenario.name}: TTS failed — {exc}")
                continue
            a = seg(scenario.baseline, 0, 4)
            b = seg(scenario.error_side, 0.1, 3.9, engine="deepgram")
            segment = AlignedSegment(
                idx=0, start=0, end=4, engine_a=a, engine_b=b, agreement=1.0,
            )
            request = JudgeRequest(segment, audio_bytes=wav.read_bytes(), mime_type="audio/wav")
            verdict = judge.judge(request)
            got = verdict.classification if verdict else "no_verdict"
            expected = scenario.expected_a2_category or "accurate"
            ok = "OK" if got == expected else "MISS"
            results[ok] += 1
            print(f"  [A2] {scenario.name:24} expected={expected:12} got={got:12} {ok}")
    return results


def evaluate_approach1() -> Counter:
    """Run the real DeepSeek judge over the strict scenarios (text-based)."""
    try:
        from approach_1 import config
    except ImportError:
        print("  (skipped: approach_1 not importable)")
        return Counter()
    if not config.DEEPSEEK_API_KEY:
        print("  (skipped: DEEPSEEK_API_KEY not set)")
        return Counter()

    from approach_1.src.evaluate import evaluate

    results: Counter = Counter()
    # Expected category uses the A1 short name; the judge emits the suffixed form.
    _TO_FULL = {
        "incorrect": "incorrect_information",
        "missing": "missing_information",
        "conflict": "conflicting_information",
        "hallucinated": "hallucinated_information",
    }
    for scenario in _STRICT:
        expected_full = _TO_FULL.get(scenario.expected_a1_category) if scenario.expected_a1_category else None
        expected = expected_full or "match"
        try:
            report = evaluate(scenario.baseline, scenario.error_side, judge=config.get_judge())
            cats = [
                k
                for k in (
                    "missing_information",
                    "incorrect_information",
                    "conflicting_information",
                    "hallucinated_information",
                )
                if getattr(report, k)
            ]
            got = cats[0] if cats else "match"
        except Exception as exc:
            print(f"  [A1] {scenario.name}: error — {type(exc).__name__}: {exc}")
            continue
        ok = "OK" if got == expected else "MISS"
        results[ok] += 1
        print(f"  [A1] {scenario.name:24} expected={expected:24} got={got:24} {ok}")
    return results


def _summarize(results: Counter) -> bool:
    total = results["OK"] + results["MISS"]
    if total == 0:
        print("No scenarios evaluated (missing keys / dependencies).")
        return True
    rate = results["OK"] / total
    print(f"\nLayer 3 aggregate category-accuracy: {results['OK']}/{total} = {rate:.0%}")
    return rate >= ACCURACY_FLOOR


def main() -> int:
    parser = argparse.ArgumentParser(description="Layer 3: real-LLM catalog evaluation")
    parser.add_argument("--approach", type=int, choices=[1, 2], help="run one approach only")
    args = parser.parse_args()

    ok = True
    if args.approach in (None, 2):
        print("Approach 2 — Gemini judge over gold-audio scenarios")
        ok &= _summarize(evaluate_approach2())
    if args.approach in (None, 1):
        print("\nApproach 1 — DeepSeek judge over gold/candidate scenarios")
        ok &= _summarize(evaluate_approach1())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())