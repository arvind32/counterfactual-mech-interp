"""Restore presence-penalty history when a generated trace prefix becomes input.

vLLM's built-in presence penalty tracks tokens generated *after* a request starts.
For a continuation experiment, the earlier generated tokens are supplied as prompt
tokens and would otherwise be forgotten by that penalty.  This processor adds the
missing penalty for those earlier tokens, while avoiding a double penalty once a
token appears in the new continuation.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)


PREFIX_IDS_KEY = "trace_prefix_presence_ids"
PENALTY_KEY = "trace_prefix_presence_penalty"


class PrefixPresenceRequestProcessor:
    """Add only the part of presence penalty missing from vLLM's native state."""

    def __init__(self, prefix_ids: list[int], penalty: float) -> None:
        self.prefix_ids = tuple(sorted(set(prefix_ids)))
        self.prefix_set = set(self.prefix_ids)
        self.penalty = float(penalty)
        self.processed_output_length = 0
        self.overlap_seen: set[int] = set()
        self._device: Optional[torch.device] = None
        self._prefix_tensor: Optional[torch.Tensor] = None
        self._overlap_tensor: Optional[torch.Tensor] = None
        self._overlap_dirty = False

    def _ensure_tensors(self, device: torch.device) -> None:
        if self._device != device:
            self._device = device
            self._prefix_tensor = torch.tensor(
                self.prefix_ids, dtype=torch.long, device=device
            )
            self._overlap_dirty = True
        if self._overlap_dirty:
            self._overlap_tensor = torch.tensor(
                sorted(self.overlap_seen), dtype=torch.long, device=device
            )
            self._overlap_dirty = False

    def __call__(
        self, output_ids: list[int], logits: torch.Tensor
    ) -> torch.Tensor:
        # output_ids is a live list retained by vLLM. Only inspect its new suffix.
        for token_id in output_ids[self.processed_output_length :]:
            if token_id in self.prefix_set and token_id not in self.overlap_seen:
                self.overlap_seen.add(token_id)
                self._overlap_dirty = True
        self.processed_output_length = len(output_ids)
        self._ensure_tensors(logits.device)

        if self._prefix_tensor is not None and self._prefix_tensor.numel():
            logits[self._prefix_tensor] -= self.penalty
        # Native vLLM already penalizes tokens seen in the new output. Add back our
        # contribution for the overlap so the total remains one presence penalty.
        if self._overlap_tensor is not None and self._overlap_tensor.numel():
            logits[self._overlap_tensor] += self.penalty
        return logits


class HistoryPresenceProcessor(AdapterLogitsProcessor):
    """vLLM batch adapter for :class:`PrefixPresenceRequestProcessor`."""

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        extra = params.extra_args or {}
        prefix_ids = extra.get(PREFIX_IDS_KEY)
        penalty = extra.get(PENALTY_KEY)
        if prefix_ids is None and penalty is None:
            return
        if not isinstance(prefix_ids, list) or not all(
            isinstance(token_id, int) and token_id >= 0 for token_id in prefix_ids
        ):
            raise ValueError(f"{PREFIX_IDS_KEY} must be a list of non-negative ints")
        if not isinstance(penalty, (int, float)) or penalty < 0:
            raise ValueError(f"{PENALTY_KEY} must be a non-negative number")

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self, params: SamplingParams
    ) -> Optional[RequestLogitsProcessor]:
        self.validate_params(params)
        extra: dict[str, Any] = params.extra_args or {}
        prefix_ids = extra.get(PREFIX_IDS_KEY)
        if not prefix_ids:
            return None
        return PrefixPresenceRequestProcessor(
            prefix_ids=prefix_ids,
            penalty=float(extra[PENALTY_KEY]),
        )
