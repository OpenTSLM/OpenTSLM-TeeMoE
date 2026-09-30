"""The aggregation expert's numerical connector and forecast decoder.

The connector summarizes the 13 candidate distributions and the reference
forecast at 8 horizon knots as input embeddings for the backbone. The decoder
reads the backbone's final 8 states and applies a bounded location correction
to the reference forecast (paper Eq. 1, Appendix A.3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

CANDIDATES = 13
QUANTILES = 9
KNOTS = 8
FEATURES = 53
HEAD_CHANNELS = 14


def sorted_quantiles(candidates: torch.Tensor) -> torch.Tensor:
    if candidates.ndim != 4 or candidates.shape[1] != CANDIDATES or candidates.shape[-1] != QUANTILES:
        raise ValueError("candidates must have shape [batch, 13, horizon, 9]")
    if not bool(torch.isfinite(candidates).all()):
        raise FloatingPointError("candidate forecasts contain nonfinite values")
    return torch.sort(candidates.float(), dim=-1, stable=True).values


def robust_scale(candidates: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """s_t = max(median candidate disagreement, reference quantile spacing, floor)."""
    ref = reference.float()
    median = ref[..., 4]
    disagreement = (sorted_quantiles(candidates)[..., 4].permute(0, 2, 1)
                    - median[..., None]).abs().median(-1).values
    spacing = torch.diff(ref, dim=-1).clamp_min(0).median(-1).values
    floor = torch.maximum(median.abs() * 1e-4, torch.full_like(median, 1e-3))
    return torch.maximum(torch.maximum(disagreement, spacing), floor)


def knot_features(candidates: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The 53-value descriptor per horizon step and the scale s_t."""
    q = sorted_quantiles(candidates)
    ref = reference.float()
    scale = robust_scale(q, ref).clamp_min(1e-6)
    median = ref[..., 4]
    horizon = candidates.shape[2]
    position = torch.linspace(0, 1, horizon, dtype=torch.float32, device=candidates.device)
    position = torch.stack((position, torch.sin(2 * math.pi * position), torch.cos(2 * math.pi * position),
                            torch.full_like(position, math.log(max(horizon, 1)))), dim=-1)
    feature = torch.cat((
        (ref - median[..., None]) / scale[..., None],
        (q[..., 4].permute(0, 2, 1) - median[..., None]) / scale[..., None],
        (q[..., 8] - q[..., 0]).permute(0, 2, 1) / scale[..., None],
        (q[..., 8] + q[..., 0] - 2 * q[..., 4]).permute(0, 2, 1) / scale[..., None],
        position[None].expand(len(q), -1, -1),
        torch.log1p(scale).clamp_max(90)[..., None],
    ), dim=-1).clamp(-20, 20)
    return feature, scale


class CandidateSummary(nn.Module):
    """Per-candidate quantile offsets and their core-8 / all-13 mean and spread."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.lattice_project = nn.Sequential(nn.Linear(QUANTILES, hidden_size), nn.LayerNorm(hidden_size))
        self.local_summary_project = nn.Sequential(nn.Linear(4 * QUANTILES, hidden_size),
                                                   nn.LayerNorm(hidden_size))
        self.horizon = nn.Parameter(torch.randn(KNOTS, hidden_size) * 0.02)

    def forward(self, offsets: torch.Tensor) -> torch.Tensor:  # [batch, 13, knots, 9]
        lattice = self.lattice_project(offsets).permute(0, 2, 1, 3) + self.horizon[None, :, None]
        core = offsets[:, :8]
        groups = torch.stack((core.mean(1), core.std(1, unbiased=False),
                              offsets.mean(1), offsets.std(1, unbiased=False)), dim=1)
        groups = groups.permute(0, 2, 1, 3).reshape(len(offsets), KNOTS, 4 * QUANTILES)
        return lattice.mean(2) + self.local_summary_project(groups)


class AggregationEditor(nn.Module):
    """Numerical connector (input embeddings) and forecast decoder (14-channel head)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(FEATURES, hidden_size), nn.LayerNorm(hidden_size))
        self.prefix = CandidateSummary(hidden_size)
        self.head = nn.Linear(hidden_size, HEAD_CHANNELS)
        nn.init.zeros_(self.head.weight)  # start exactly at the reference forecast
        nn.init.zeros_(self.head.bias)

    def forward(self, backbone: nn.Module, candidates: torch.Tensor,
                reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Refined quantiles [batch, horizon, 9] and the ungated normalized correction d_t."""
        batch, _, horizon, _ = candidates.shape
        if reference.shape != (batch, horizon, QUANTILES):
            raise ValueError("reference must have shape [batch, horizon, 9]")
        knot = torch.linspace(0, horizon - 1, KNOTS, device=candidates.device).round().long()
        cand_k, ref_k = candidates.index_select(2, knot), reference.index_select(1, knot)
        feature, scale = knot_features(cand_k, ref_k)
        offsets = ((sorted_quantiles(cand_k) - ref_k[:, None].float())
                   / scale[:, None, :, None]).clamp(-20, 20)
        dtype = next(backbone.parameters()).dtype
        query = self.projector(feature).to(dtype) + self.prefix(offsets).to(dtype)
        order = torch.arange(KNOTS, device=candidates.device)
        # Evidence in reverse then forward order, then the output queries: every output
        # position can attend to the whole horizon under causal attention.
        sequence = torch.cat((query[:, order.flip(0)], query, query), dim=1)
        states = backbone(inputs_embeds=sequence, attention_mask=torch.ones(sequence.shape[:2],
                          dtype=torch.long, device=sequence.device),
                          position_ids=torch.arange(sequence.shape[1], device=sequence.device)[None]
                          .expand(batch, -1), use_cache=False, return_dict=True).last_hidden_state[:, -KNOTS:]
        raw = self.head(states.float())
        if horizon != KNOTS:
            raw = F.interpolate(raw.permute(0, 2, 1), size=horizon, mode="linear",
                                align_corners=True).permute(0, 2, 1)
        weights = torch.softmax(raw[..., 1:], dim=-1)
        medians = sorted_quantiles(candidates)[..., 4].permute(0, 2, 1)
        ref_median = reference[..., 4].float()
        scale = robust_scale(candidates, reference).clamp_min(1e-6)
        comparison = (weights * medians).sum(-1)
        correction = 0.5 * torch.tanh(raw[..., 0]) * torch.tanh((comparison - ref_median) / scale)
        shifted = ref_median + correction * scale
        return reference.float() + (shifted - ref_median)[..., None], correction

    def predict(self, backbone: nn.Module, candidates: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Inference forecast with the disagreement gate g_t (paper Eq. 6)."""
        _, correction = self(backbone, candidates, reference)
        ref = reference.float()
        median = ref[..., 4]
        disagreement = (sorted_quantiles(candidates)[..., 4].permute(0, 2, 1)
                        - median[..., None]).abs().median(-1).values
        floor = torch.maximum(median.abs() * 1e-4, torch.full_like(median, 1e-3))
        variance = torch.maximum((ref[..., 6] - ref[..., 2]).clamp_min(0), floor).square()
        gate = variance / (variance + disagreement.square())
        shifted = median + gate * correction.float() * robust_scale(candidates, ref)
        return ref + (shifted - median)[..., None]


@dataclass
class EditorLoss:
    total: torch.Tensor  # sum over rows of the per-row loss
    rows: float


def editor_loss(correction: torch.Tensor, *, target: torch.Tensor, reference: torch.Tensor,
                candidates: torch.Tensor, knot_weight: torch.Tensor, beta: float = 0.05,
                edit_weight: float = 0.2) -> EditorLoss:
    """Knot-weighted SmoothL1 to the clipped normalized target plus a squared-edit penalty (Eq. 8)."""
    scale = robust_scale(candidates, reference)
    target_d = ((target.float() - reference[..., 4].float()) / scale).clamp(-0.5, 0.5)
    weights = knot_weight.float() / knot_weight.float().sum(1, keepdim=True).clamp_min(1e-12)
    fit = F.smooth_l1_loss(correction.float(), target_d, reduction="none", beta=beta)
    per_row = (fit * weights).sum(1) + edit_weight * (correction.float().square() * weights).sum(1)
    return EditorLoss(per_row.sum(), float(len(correction)))
