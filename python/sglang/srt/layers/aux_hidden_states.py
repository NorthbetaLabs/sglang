"""Aux hidden states captured for Eagle3/DFlash draft models."""

from typing import List, Optional, Union

import torch

# Two representations coexist: models migrated to AuxHiddenStatePacker pass one
# packed [tokens, K * hidden] tensor, the rest still pass a list of K tensors.
AuxHiddenStates = Union[torch.Tensor, List[torch.Tensor]]


class AuxHiddenStateList(list):
    """Retain aux snapshots independently of storage reused by later layers.

    Views and reusable communication buffers are borrowed even when they
    are different tensor objects from the source.
    """

    # ``capture`` clones borrowed values, so a later layer may reuse storage.
    copies_on_append = False

    def capture(self, hidden: torch.Tensor, *, owned: bool = False) -> None:
        """Copy borrowed storage, or adopt a value whose ownership is transferred."""
        self.append(hidden if owned else hidden.clone())

    def append(self, hidden: torch.Tensor) -> None:
        if getattr(self, "copies_on_append", False):
            hidden = hidden.clone()
        super().append(hidden)


class AuxHiddenStatePacker:
    """Drop-in for the ``[]`` a model collects Eagle3/DFlash captures into.

    Each ``.append()`` writes into one preallocated ``[tokens, K * hidden]``
    buffer, avoiding the list path's transient ~2x HBM at ``torch.cat``.
    Assumes all captures share leading shape and feature size.
    """

    # ``append`` copies, so producers need not clone a tensor they later mutate.
    copies_on_append = True

    def __init__(self, num_captures: int) -> None:
        self._num_captures = int(num_captures)
        self._buffer: Optional[torch.Tensor] = None
        self._feature_size: Optional[int] = None
        self._idx = 0

    def append(self, hidden: torch.Tensor) -> None:
        feature_size = int(hidden.shape[-1])
        if self._buffer is None:
            self._feature_size = feature_size
            self._buffer = hidden.new_empty(
                (*hidden.shape[:-1], feature_size * self._num_captures)
            )
        start = self._idx * self._feature_size
        self._buffer[..., start : start + self._feature_size].copy_(hidden)
        self._idx += 1

    def __len__(self) -> int:
        return self._idx

    def finalize(self) -> torch.Tensor:
        """Return the packed buffer; callers guard the empty case on ``len()``."""
        assert (
            self._buffer is not None and self._idx == self._num_captures
        ), f"captured {self._idx} of {self._num_captures} aux hidden states"
        return self._buffer


# What a model hands down the capture path: a plain list, or a packer writing in place.
AuxHiddenStateAccumulator = Union[AuxHiddenStateList, List[torch.Tensor], AuxHiddenStatePacker]


def pack_aux_hidden_states(aux_hidden_states: AuxHiddenStates) -> torch.Tensor:
    if isinstance(aux_hidden_states, torch.Tensor):
        return aux_hidden_states
    return torch.cat(aux_hidden_states, dim=-1)
