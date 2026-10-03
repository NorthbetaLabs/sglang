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


def distributed_final_sample(
    target_probs_shard,      # [bs, slots, V/tp] softmax over the shard
    draft_probs_shard,       # [bs, slots, V/tp] sparse at shard-local candidates
    cur_prob_rows,           # [bs] residual base row per request
    all_drafts_accepted,     # [bs] bool: pure-target branch flag
    uniform_final,           # [bs]
    group,                   # tp process group (gloo/nccl all_reduce here)
):
    """Final bonus-token sampling with sharded vocab — exact equivalent of
    the classic kernel's Pass1+Pass2, with O(1) communication.

    Protocol:
      1. per-rank fragment residual sum (kernel)
      2. all_reduce the per-rank sums -> global residual norm Z
      3. uniform*Z -> target_u; per-rank prefix from the gathered sums
         decides whether THIS rank's shard contains the landing point
      4. landing rank scans its local CDF (kernel); others yield -1
      5. all_reduce MAX of the located token -> global token id
    """
    from sglang.kernels.ops.speculative.reject_sampling import (
        fragment_residual_sum,
        fragment_final_locate,
    )

    bs = target_probs_shard.shape[0]
    dev = target_probs_shard.device

    # 1. per-rank residual fragment sums
    # For the pure-target branch (all accepted) residual = p itself; the
    # draft shard is zeroed so the kernel's (p-q)+ degenerates to p.
    frag_sum = fragment_residual_sum(
        target_probs_shard,
        draft_probs_shard
        if not all_drafts_accepted.all()
        else torch.zeros_like(draft_probs_shard),
        cur_prob_rows,
    )  # [bs]

    # 2. global residual norm = sum over ranks (exact: denominator additive)
    global_z = frag_sum.clone()
    torch.distributed.all_reduce(global_z, op=torch.distributed.ReduceOp.SUM, group=group)

    # 3. per-rank prefix: sum of fragment sums of ranks BEFORE me.
    #    all_gather the per-rank sums, prefix on host.
    world = torch.distributed.get_world_size(group)
    per_rank = list(torch.empty(world, device=dev).chunk(world)) if False else None
    gathered = torch.empty((world,) + frag_sum.shape, device=dev)
    torch.distributed.all_gather_into_tensor(gathered, frag_sum, group=group)
    rank = torch.distributed.get_rank(group)
    prefix = gathered[:rank].sum(dim=0) if rank > 0 else torch.zeros_like(frag_sum)

    target_u = uniform_final.to(dev) * global_z
    # landing offset inside my shard (only meaningful if the landing point is mine)
    local_target_u = target_u - prefix

    # 4. landing rank scan: rank r owns [prefix_r, prefix_r + frag_r).
    #    A rank lands iff target_u in [prefix_r, prefix_r+frag_r).
    lower = prefix
    upper = prefix + frag_sum
    is_landing = (target_u >= lower) & (target_u < upper)
    offsets = torch.where(
        is_landing, local_target_u, torch.full_like(local_target_u, float("inf"))
    )

    found_local = fragment_final_locate(
        target_probs_shard,
        draft_probs_shard
        if not all_drafts_accepted.all()
        else torch.zeros_like(draft_probs_shard),
        cur_prob_rows,
        offsets,
    )  # [bs] shard-local id or -1

    # 5. combine: landing rank contributes its local id + shard base offset.
    vocab_start = target_probs_shard.shape[-1] * rank  # contiguous shard assumption
    token_global = torch.where(
        found_local >= 0, found_local + vocab_start, torch.zeros_like(found_local)
    )
    final_token = token_global.clone()
    torch.distributed.all_reduce(final_token, op=torch.distributed.ReduceOp.MAX, group=group)
    # Degenerate residual (Z == 0): classic falls back to VOCAB-1. The MAX
    # reduce over zeros yields 0; caller maps that case explicitly.
    return final_token, global_z
