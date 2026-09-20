"""Summarisation metrics for discharge summaries and similar (FR-X-05): ROUGE and
BERTScore.

Both compare a *candidate* summary (what a model wrote) with a *reference* (what a
person wrote) and report precision, recall and F1. They measure **similarity to a
reference, not correctness**: a fluent summary that omits the one abnormal result can
score well, and a faithful paraphrase can score badly. Use them to compare systems on
your own data, beside the groundedness checks in :mod:`openbtk.eval.groundedness`, never
as a safety measure.

**ROUGE** counts overlapping words (``rouge1``), word pairs (``rouge2``) and the longest
common subsequence (``rougeL``). It wraps Google's ``rouge-score`` (the ``eval`` extra).
Its tokenizer lowercases and keeps only ``a-z`` and ``0-9``, so a letter outside ASCII,
or a symbol such as ``%`` or ``>``, is dropped before scoring. That is a property of the
library and a real limit for clinical text; it is why stemming is on by default (the
usual reporting convention) but off-by-choice is one argument away.

**BERTScore** (Zhang et al., 2020) matches each candidate token to its most similar
reference token in a language model's embedding space, so it credits a paraphrase that
ROUGE misses. It is implemented here on ``transformers`` instead of wrapping the
``bert-score`` package, because that package fetches its model by name with no way to
pin a revision, and an unpinned download is not something an auditable pipeline should
do (security review finding S-3). **Both the model and its commit are required, and so
is the layer**: OpenBTK does not pick a model or a layer for you, because the right ones
are a choice you should be able to defend. The implementation follows the reference
greedy matching and was checked against the ``bert-score`` package (see the tests); it
does not support idf weighting or baseline rescaling, so its numbers are the *raw*
scores, comparable only with other raw scores from the same model and layer.

Reports hold **scores and counts only, never text**, so they are safe to log and to
attach to a manifest (:func:`summarisation_manifest`) or a model card.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from openbtk.core._lazy import require
from openbtk.core.errors import ConfigError, ProcessingError
from openbtk.core.provenance import ComponentProvenance, ModelIdentity
from openbtk.core.schemas import JsonValue  # noqa: TC001
from openbtk.eval.manifest import EvalManifest, build_manifest

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from openbtk.core.provenance import DataDigest

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")
"""The ROUGE variants offered. (``rougeLsum`` is left out: it needs sentence splitting
that this project does not do for clinical text.)"""


class SummaryPair(BaseModel):
    """One candidate summary and the reference it is compared with.

    Example:
        >>> SummaryPair(
        ...     example_id="d1", reference="Stable on discharge.", candidate="Stable."
        ... ).example_id
        'd1'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str = Field(..., min_length=1)
    reference: str
    candidate: str


class Prf(BaseModel):
    """Precision, recall and F1, each in ``[0, 1]``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    precision: float
    recall: float
    f1: float


class ExampleScores(BaseModel):
    """The scores of one example, keyed by its id. No text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str
    rouge: dict[str, Prf] = Field(default_factory=dict)
    bertscore: Prf | None = None


class SummarisationReport(BaseModel):
    """Mean scores over the examples (a macro average) and each example's own.

    Example:
        >>> report = evaluate_summaries([], rouge_types=())
        >>> report.n
        0
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int = Field(..., ge=0, description="Examples scored.")
    rouge: dict[str, Prf] = Field(default_factory=dict)
    bertscore: Prf | None = None
    examples: list[ExampleScores] = Field(default_factory=list)
    started_at: datetime
    ended_at: datetime

    def as_dict(self) -> dict[str, JsonValue]:
        """A flat ``{name: number}`` view (``rouge1_f1``, ``bertscore_f1``, ``n``, ...),
        the shape ``EvalManifest.report`` and ``ModelCard.with_evaluation`` read."""
        flat: dict[str, JsonValue] = {"n": self.n}
        for name, prf in self.rouge.items():
            flat[f"{name}_precision"] = prf.precision
            flat[f"{name}_recall"] = prf.recall
            flat[f"{name}_f1"] = prf.f1
        if self.bertscore is not None:
            flat["bertscore_precision"] = self.bertscore.precision
            flat["bertscore_recall"] = self.bertscore.recall
            flat["bertscore_f1"] = self.bertscore.f1
        return flat


# -------------------------------------------------------------------------- ROUGE


def rouge_scores(
    reference: str,
    candidate: str,
    *,
    types: Sequence[str] = ROUGE_TYPES,
    use_stemmer: bool = True,
) -> dict[str, Prf]:
    """ROUGE precision, recall and F1 of one candidate against one reference.

    Args:
        reference: The reference summary.
        candidate: The candidate summary.
        types: Which variants: ``rouge1``, ``rouge2``, ``rougeL``.
        use_stemmer: Porter-stem words first (the usual reporting convention).

    Raises:
        ConfigError: On an unknown variant.
        MissingDependencyError: If ``rouge-score`` is not installed (``eval`` extra).

    Example:
        >>> scores = rouge_scores("the patient is stable", "the patient is stable")
        >>> scores["rouge1"].f1
        1.0
    """
    unknown = [t for t in types if t not in ROUGE_TYPES]
    if unknown:
        raise ConfigError(
            f"Unknown ROUGE variant(s) {unknown}; choose from {list(ROUGE_TYPES)}.",
            context={"variants": unknown},
        )
    if not types:
        return {}
    rouge_scorer = require("rouge_score.rouge_scorer", extra="eval")
    scorer = rouge_scorer.RougeScorer(list(types), use_stemmer=use_stemmer)
    raw = scorer.score(reference, candidate)  # (target, prediction)
    return {
        name: Prf(
            precision=float(raw[name].precision),
            recall=float(raw[name].recall),
            f1=float(raw[name].fmeasure),
        )
        for name in types
    }


# ---------------------------------------------------------------------- BERTScore


class BertScorer:
    """BERTScore with a model, revision and layer you choose. See the module docstring.

    Args:
        model: A Hugging Face Hub encoder id (or local path).
        revision: The pinned commit of ``model``. Required: an unpinned download follows
            a moving branch.
        layer: Which hidden layer's token embeddings to compare, ``1`` to the model's
               number of layers (``0`` would be the input embeddings, which carry no
               context). The BERTScore paper tuned one per model; choose and record
               yours.
        device: A torch device string.
        max_length: Tokens kept per text; a longer text is truncated, and the score
                    covers only what was kept.

    Example:
        >>> scorer = BertScorer(model="org/encoder", revision="abc1234", layer=8)
        >>> scorer.provenance().model_identity.revision  # nothing is loaded yet
        'abc1234'
    """

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        layer: int,
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        if not revision:
            raise ConfigError(
                "BERTScore needs a pinned model revision.",
                context={"component": "eval.bertscore"},
            )
        if layer < 1 or max_length < 3:
            raise ConfigError(
                "layer must be at least 1 and max_length at least 3.",
                context={"component": "eval.bertscore"},
            )
        self._model_name = model
        self._revision = revision
        self._layer = layer
        self._device = device
        self._max_length = max_length
        self._loaded: tuple[Any, Any, Any] | None = None

    def provenance(self) -> ComponentProvenance:
        """The model, revision and layer, for a manifest."""
        return ComponentProvenance(
            registry_key="eval.bertscore",
            class_name=type(self).__name__,
            package_version=_openbtk_version(),
            config={
                "model": self._model_name,
                "layer": self._layer,
                "device": self._device,
                "max_length": self._max_length,
            },
            model_identity=ModelIdentity(
                name=self._model_name, revision=self._revision, source="huggingface"
            ),
        )

    def _load(self) -> tuple[Any, Any, Any]:
        if self._loaded is None:
            transformers = require("transformers", extra="text")
            torch = require("torch", extra="llms")
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self._model_name, revision=self._revision
            )
            model = transformers.AutoModel.from_pretrained(
                self._model_name, revision=self._revision
            )
            model.to(self._device)
            model.eval()
            layers = int(getattr(model.config, "num_hidden_layers", 0))
            if layers and self._layer > layers:
                raise ConfigError(
                    f"layer {self._layer} is beyond the model's {layers} layers.",
                    context={"component": "eval.bertscore"},
                )
            self._loaded = (tokenizer, model, torch)
        return self._loaded

    def _embed(self, text: str) -> tuple[Any, Any]:
        """Unit-length embeddings of every token, and which are real (not special)."""
        tokenizer, model, torch = self._load()
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        special = encoded.pop("special_tokens_mask")[0].bool()
        encoded = {k: v.to(self._device) for k, v in encoded.items()}
        with torch.no_grad():
            hidden = model(**encoded, output_hidden_states=True).hidden_states[
                self._layer
            ][0]
        norms = torch.linalg.norm(hidden, dim=-1, keepdim=True).clamp_min(1e-12)
        return hidden / norms, (~special).to(hidden.device)

    def score(self, reference: str, candidate: str) -> Prf:
        """Raw BERTScore precision, recall and F1 of ``candidate`` vs ``reference``.

        As in the reference implementation, the special tokens (``[CLS]``, ``[SEP]``)
        carry zero weight in the averages but still take part in the matching, so a
        token can be matched to the other text's special token. An empty (or
        all-special-token) text scores ``0`` on all three.

        Raises:
            ProcessingError: If the model fails. The message never contains the text.
        """
        try:
            ref, ref_real = self._embed(reference)
            hyp, hyp_real = self._embed(candidate)
            if not bool(ref_real.any()) or not bool(hyp_real.any()):
                return Prf(precision=0.0, recall=0.0, f1=0.0)
            similarity = hyp @ ref.T  # (candidate tokens, reference tokens)
            precision = float(similarity.max(dim=1).values[hyp_real].mean())
            recall = float(similarity.max(dim=0).values[ref_real].mean())
        except (ConfigError, ProcessingError):
            raise
        except Exception as e:  # torch / transformers raise their own hierarchy
            raise ProcessingError(
                "BERTScore failed.",
                context={"component": "eval.bertscore", "model": self._model_name},
            ) from e
        denominator = precision + recall
        f1 = 2 * precision * recall / denominator if denominator else 0.0
        return Prf(precision=precision, recall=recall, f1=f1)


def _openbtk_version() -> str:
    from openbtk.core.base import _openbtk_version as version

    return version()


# --------------------------------------------------------------------- evaluation


def evaluate_summaries(
    pairs: Iterable[SummaryPair],
    *,
    rouge_types: Sequence[str] = ROUGE_TYPES,
    use_stemmer: bool = True,
    bertscorer: BertScorer | None = None,
) -> SummarisationReport:
    """Score candidate summaries against references.

    Streams the pairs, holding only the running sums and one small record of scores per
    example. The report's means are macro averages: every example counts once, however
    long its summary.

    Args:
        pairs: The candidate and reference of each example.
        rouge_types: ROUGE variants to compute (``()`` skips ROUGE).
        use_stemmer: Stem before ROUGE matching.
        bertscorer: Also compute BERTScore with this scorer (``None`` skips it).

    Example:
        >>> report = evaluate_summaries(
        ...     [SummaryPair(example_id="a", reference="stable", candidate="stable")]
        ... )
        >>> report.rouge["rouge1"].f1, report.n
        (1.0, 1)
    """
    started = datetime.now(UTC)
    rouge_sums: dict[str, list[float]] = {t: [0.0, 0.0, 0.0] for t in rouge_types}
    bert_sums = [0.0, 0.0, 0.0]
    examples: list[ExampleScores] = []
    for pair in pairs:
        rouge = rouge_scores(
            pair.reference, pair.candidate, types=rouge_types, use_stemmer=use_stemmer
        )
        for name, prf in rouge.items():
            rouge_sums[name][0] += prf.precision
            rouge_sums[name][1] += prf.recall
            rouge_sums[name][2] += prf.f1
        bert = None
        if bertscorer is not None:
            bert = bertscorer.score(pair.reference, pair.candidate)
            bert_sums[0] += bert.precision
            bert_sums[1] += bert.recall
            bert_sums[2] += bert.f1
        examples.append(
            ExampleScores(example_id=pair.example_id, rouge=rouge, bertscore=bert)
        )
    n = len(examples)

    def mean(total: list[float]) -> Prf:
        return Prf(
            precision=total[0] / n if n else 0.0,
            recall=total[1] / n if n else 0.0,
            f1=total[2] / n if n else 0.0,
        )

    return SummarisationReport(
        n=n,
        rouge={name: mean(total) for name, total in rouge_sums.items()} if n else {},
        bertscore=mean(bert_sums) if bertscorer is not None and n else None,
        examples=examples,
        started_at=started,
        ended_at=datetime.now(UTC),
    )


def summarisation_manifest(
    report: SummarisationReport,
    *,
    bertscorer: BertScorer | None = None,
    input_digests: Sequence[DataDigest] = (),
) -> EvalManifest:
    """The provenance record for a summarisation run: when, the mean scores and counts,
    the BERTScore model and revision if one was used, and the digests of the input
    files. Never the summaries."""
    return build_manifest(
        "summarisation",
        report.as_dict(),
        started_at=report.started_at,
        ended_at=report.ended_at,
        component=bertscorer.provenance() if bertscorer is not None else None,
        input_digests=input_digests,
    )
