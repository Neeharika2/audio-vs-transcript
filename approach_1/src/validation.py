"""Validation harness: synthetic error injection + detector precision/recall.

Generates candidate transcripts with known injected errors (ground truth
annotations), runs the evaluation pipeline, and measures how well the
detector recovers each category (missing / incorrect / conflicting /
hallucinated). Use this as a regression suite when changing prompts or models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz

from approach_1.src.align import segment_sentences
from approach_1.src.evaluate import evaluate
from approach_1.src.models import ErrorItem
from approach_1.src.normalize import normalize_text

CATEGORIES = [
    "missing_information",
    "incorrect_information",
    "conflicting_information",
    "hallucinated_information",
]


@dataclass
class Annotation:
    category: str
    reference_text: str | None = None
    generated_text: str | None = None


@dataclass
class InjectedCase:
    gold: str
    candidate: str
    annotations: list[Annotation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Error injection
# ---------------------------------------------------------------------------

def build_case(gold_sentences: list[str], keep: list[int], mutate: dict | None = None,
               conflict: list[int] | None = None, insert: list[tuple[int, str]] | None = None) -> InjectedCase:
    """Build a synthetic candidate from gold sentences.

    keep: indices of sentences that remain unchanged in the candidate.
    mutate: {idx: replacement_text} -> incorrect_information (replaces the kept sentence).
    conflict: indices to directly contradict -> conflicting_information (replaces the kept sentence).
    insert: [(idx, text)] sentences inserted after position idx -> hallucinated_information.
    """
    gold = " ".join(gold_sentences)
    mutate = mutate or {}
    conflict = conflict or []
    insert = insert or []

    present = [
        i for i in range(len(gold_sentences))
        if i in keep or i in mutate or i in conflict
    ]
    candidate_parts: list[str] = []
    generated_for: dict[int, str] = {}
    for i in present:
        if i in mutate:
            text = mutate[i]
        elif i in conflict:
            text = _contradict(gold_sentences[i])
        else:
            text = gold_sentences[i]
        candidate_parts.append(text)
        generated_for[i] = text

    annotations: list[Annotation] = []
    for i in mutate:
        annotations.append(
            Annotation(category="incorrect_information",
                       reference_text=gold_sentences[i], generated_text=mutate[i])
        )
    for i in conflict:
        annotations.append(
            Annotation(category="conflicting_information",
                       reference_text=gold_sentences[i], generated_text=generated_for[i])
        )
    for idx, text in insert:
        candidate_parts.insert(idx, text)
        annotations.append(
            Annotation(category="hallucinated_information", generated_text=text)
        )
    for i in range(len(gold_sentences)):
        if i not in present:
            annotations.append(
                Annotation(category="missing_information", reference_text=gold_sentences[i])
            )

    return InjectedCase(gold=gold, candidate=" ".join(candidate_parts), annotations=annotations)


def _contradict(sentence: str) -> str:
    """Produce a direct contradiction of a sentence."""
    lowered = sentence
    for prefix in ("The ", "He ", "She ", "It ", "They "):
        if lowered.startswith(prefix):
            return "NOT " + lowered
    return "NOT " + lowered


# ---------------------------------------------------------------------------
# Matching detected findings against annotations
# ---------------------------------------------------------------------------

def _evidence(item: ErrorItem) -> str:
    return (item.reference_text or item.generated_text or "")


def _matches(text_a: str, text_b: str, threshold: float = 0.8) -> bool:
    if not text_a or not text_b:
        return False
    if normalize_text(text_a) == normalize_text(text_b):
        return True
    return fuzz.token_set_ratio(normalize_text(text_a), normalize_text(text_b)) / 100.0 >= threshold


def _match_findings(findings: list[ErrorItem], annotations: list[Annotation]) -> dict:
    matched_findings: set[int] = set()
    matched_annots: set[int] = set()
    for f_idx, finding in enumerate(findings):
        ev = _evidence(finding)
        for a_idx, ann in enumerate(annotations):
            if a_idx in matched_annots or ann.category != finding.category:
                continue
            ann_ev = ann.reference_text or ann.generated_text or ""
            if _matches(ev, ann_ev):
                matched_findings.add(f_idx)
                matched_annots.add(a_idx)
                break
    return {"matched_findings": matched_findings, "matched_annots": matched_annots}


def evaluate_case(evaluate_fn, case: InjectedCase) -> tuple[list[ErrorItem], list[Annotation]]:
    """Run the pipeline on a case and return (findings, annotations)."""
    report = evaluate_fn(case.gold, case.candidate)
    findings: list[ErrorItem] = (
        report.missing_information
        + report.incorrect_information
        + report.conflicting_information
        + report.hallucinated_information
    )
    return findings, case.annotations


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[tuple[list[ErrorItem], list[Annotation]]]) -> dict:
    """Per-category precision / recall / F1 plus overall aggregates."""
    metrics: dict = {}
    for category in CATEGORIES:
        findings = [f for found, _ in results for f in found if f.category == category]
        annots = [a for _, anns in results for a in anns if a.category == category]
        total_found = len(findings)
        total_injected = len(annots)

        used_annots: set[int] = set()
        for f in findings:
            ev = _evidence(f)
            for a_idx, ann in enumerate(annots):
                if a_idx in used_annots:
                    continue
                if _matches(ev, ann.reference_text or ann.generated_text or ""):
                    used_annots.add(a_idx)
                    break
        tp = len(used_annots)
        fp = total_found - tp
        fn = total_injected - tp

        precision = tp / total_found if total_found else 1.0
        recall = tp / total_injected if total_injected else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics[category] = {
            "injected": total_injected,
            "detected": tp,
            "false_positives": fp,
            "missed": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    all_injected = sum(m["injected"] for m in metrics.values())
    all_detected = sum(m["detected"] for m in metrics.values())
    all_fp = sum(m["false_positives"] for m in metrics.values())
    precision = all_detected / max(1, all_detected + all_fp)
    recall = all_detected / max(1, all_injected)
    metrics["overall"] = {
        "injected": all_injected,
        "detected": all_detected,
        "false_positives": all_fp,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0, 3),
    }
    return metrics


def run_suite(cases: list[InjectedCase] | None = None, evaluate_fn=None) -> dict:
    """Run the full suite and return per-category + overall metrics."""
    cases = cases if cases is not None else default_suite()
    evaluate_fn = evaluate_fn or (lambda g, c: evaluate(g, c))
    results = [evaluate_case(evaluate_fn, case) for case in cases]
    return compute_metrics(results)


def default_suite() -> list[InjectedCase]:
    """A small deterministic suite covering each error category."""
    sents = [
        "The patient takes fifty milligrams of aspirin daily.",
        "He was admitted on Tuesday.",
        "The hospital is on Main Street.",
    ]
    cases = [
        build_case(sents, keep=[0, 1, 2]),
        build_case(sents, keep=[0, 2]),
        build_case(sents, keep=[0, 1, 2], mutate={1: "He was admitted on Thursday."}),
        build_case(sents, keep=[0, 1, 2], conflict=[1]),
        build_case(sents, keep=[0, 1, 2], insert=[(2, "He also owns a pet dragon.")]),
    ]
    return cases
