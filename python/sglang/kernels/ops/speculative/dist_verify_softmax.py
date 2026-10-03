"""Distributed verify softmax: replace the full-vocab logits all-gather
(900us/step, 19% of TPOT) with a two-stage scalar protocol.

Mathematically exact. The vocab is sharded over TP ranks; softmax's
denominator is additive; the final-sample CDF concatenates rank fragments
in vocab order, so the global CDF is the rank-ordered join of local CDFs.

Protocol per verify step:
  stage 0 (no comm): per-rank partial exp-sum of its vocab shard
  stage 1 (tiny AG): candidate logits (<=16/rank shard) + partial sums
  local: accept/reject decisions for every slot (needs only the 16
         candidate probs + global denominator)
  stage 2 (tiny AG): each rank's residual-fragment sum -> global residual
         norm; uniform-final landing rank determined locally from the
         per-rank fragment sums; the landing rank scans its local CDF
  stage 3 (tiny AG): one final token id
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

if TYPE_CHECKING:
    pass


def shard_partial_sum(logits_shard: torch.Tensor) -> torch.Tensor:
    """Per-rank exp-sum of a vocab shard. logits_shard: [T, V/tp]."""
    return torch.exp(logits_shard.float()).sum(dim=-1)  # [T]


def shard_candidate_logits(
    logits_shard: torch.Tensor,
    candidate_ids: torch.Tensor,
    vocab_start: int,
    vocab_end: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather the shard-local candidate logits and a validity mask.

    candidate_ids: [bs, slots] global vocab ids. For each id that falls
    inside [vocab_start, vocab_end) take the logit; mask=1.
    Returns (values [bs, slots] float32, mask [bs, slots]).
    """
    bs, slots = candidate_ids.shape
    local = candidate_ids - vocab_start
    inside = (candidate_ids >= vocab_start) & (candidate_ids < vocab_end)
    local_clamped = local.clamp(min=0, max=logits_shard.shape[-1] - 1)
    # logits_shard: [T, V/tp] — token dim must be broadcastable to bs.
    if logits_shard.shape[0] == 1:
        rows = logits_shard.expand(bs, -1)
    else:
        rows = logits_shard
    vals = torch.gather(rows, 1, local_clamped)
    return vals.float(), inside

# The accept-decision and final-sample kernels live in reject_sampling.py;
# see fragment_residual_sum_kernel / rank_locate_final_kernel there.
