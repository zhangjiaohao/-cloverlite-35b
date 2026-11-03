"""Low-confidence block resolution and expert coreset reference algorithms.

The caller's preview implementation MUST leave the committed prefix state unchanged.
No claim is made that the existing Strata causal engine implements this interface.
"""
from dataclasses import dataclass
import math
from typing import Callable, Protocol, Sequence


@dataclass(frozen=True)
class Prediction:
    token: int
    confidence: float


class BlockRuntime(Protocol):
    def preview(self, block: Sequence[int], pending: Sequence[bool]) -> Sequence[Prediction]:
        """Predict every position using the same committed prefix without changing it."""

    def commit(self, tokens: Sequence[int]) -> None:
        """Causally advance prefix state using a fully resolved block."""


@dataclass(frozen=True)
class BlockResult:
    tokens: tuple[int, ...]
    stopped: bool
    rounds: int


def resolve_block(runtime: BlockRuntime, *, width: int, mask_token: int, threshold: float,
                  should_stop: Callable[[], bool] = lambda: False) -> tuple[list[int], int]:
    if width <= 0 or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('invalid block size or confidence threshold')
    block = [mask_token] * width
    pending = [True] * width
    for step in range(1, width + 1):
        if should_stop():
            raise InterruptedError('block generation cancelled')
        predictions = runtime.preview(tuple(block), tuple(pending))
        if len(predictions) != width:
            raise ValueError('preview returned the wrong number of positions')
        for pos, pred in enumerate(predictions):
            if pending[pos] and (pred.token == mask_token or not math.isfinite(pred.confidence)
                                 or not 0 <= pred.confidence <= 1):
                raise ValueError('invalid prediction for an unresolved position')
        selected = [i for i, active in enumerate(pending) if active and predictions[i].confidence > threshold]
        if not selected:
            # Stable first-position tie breaking, matching argmax.
            selected = [max((i for i, active in enumerate(pending) if active),
                            key=lambda i: predictions[i].confidence)]
        for pos in selected:
            block[pos] = predictions[pos].token
            pending[pos] = False
        if not any(pending):
            return block, step
    raise RuntimeError('block still contains unresolved positions')


def generate_blocks(runtime: BlockRuntime, *, max_new_tokens: int, block_size: int = 32,
                    mask_token: int = 151669, threshold: float = 0.95,
                    eos_tokens: Sequence[int] = (248046, 248044),
                    should_stop: Callable[[], bool] = lambda: False):
    """Yield complete blocks truncated at EOS/output limit, committing only continuing blocks.

    Runtime owns prompt prefill and logits processing (penalties, sampling and mask suppression).
    Full-size preview is required even when fewer output tokens remain.
    """
    if max_new_tokens < 0 or block_size <= 0:
        raise ValueError('invalid generation limit')
    produced = 0
    while produced < max_new_tokens:
        block, rounds = resolve_block(runtime, width=block_size, mask_token=mask_token,
                                      threshold=threshold, should_stop=should_stop)
        keep = block[:min(block_size, max_new_tokens - produced)]
        stopped = False
        for i, token in enumerate(keep):
            if token in eos_tokens:
                keep = keep[:i + 1]
                stopped = True
                break
        produced += len(keep)
        continuing = not stopped and produced < max_new_tokens
        if continuing:
            runtime.commit(tuple(keep))
        yield BlockResult(tuple(keep), stopped, rounds)
        if stopped:
            return


def select_expert_coreset(logits: Sequence[Sequence[float]], valid: Sequence[bool], *,
                          top_k: int, top_p: float) -> tuple[int, ...]:
    """Return selected expert IDs; stable ties use lower IDs, like official Clover."""
    if not logits or len(logits) != len(valid):
        raise ValueError('logits and valid mask must contain the same nonzero number of rows')
    n = len(logits[0])
    if not n or not 1 <= top_k <= n or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError('invalid coreset dimensions or threshold')
    totals = [0.0] * n
    for row, use in zip(logits, valid):
        if len(row) != n or any(not math.isfinite(x) for x in row):
            raise ValueError('router logits must be finite and have equal width')
        if not use:
            continue
        m = max(row)
        weights = [math.exp(v - m) for v in row]
        denominator = sum(weights)
        for i in sorted(range(n), key=lambda i: (-row[i], i))[:top_k]:
            totals[i] += weights[i] / denominator
    mass = sum(totals)
    if mass <= 0:
        raise ValueError('coreset needs at least one valid token')
    order = sorted(range(n), key=lambda i: (-totals[i], i))
    result, previous = [], 0.0
    for i in order:
        if result and previous >= top_p:
            break
        result.append(i)
        previous += totals[i] / mass
    return tuple(result)
