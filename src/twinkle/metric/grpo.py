# Copyright (c) ModelScope Contributors. All rights reserved.
"""GRPO-specific per-token log-prob diagnostics.

``GRPOMetric`` consumes the data the GRPO training loop already passes
through ``forward_backward`` / exposes on ``ModelOutput`` \u2014 no changes to
the loss or model return contract are required:

* ``outputs['logps']``: new-policy per-token log prob
  (``selective_log_softmax(logits / temperature, labels)`` computed
  inside the model). In the Megatron + padding_free path this is a
  Python list of per-microbatch tensors of possibly different shapes;
  elsewhere it is a single ``[num_seq, max_seq_len]`` tensor.
* ``inputs['labels']``: used to derive the generated-token mask
  (``labels != ignore_index``). In the Megatron path ``inputs`` is a
  list of per-microbatch dicts (already unpacked); in the Transformers
  path it is a single unpacked dict.
* ``kwargs['old_logps']``: sampling-policy per-token log prob, piped in
  verbatim by ``OptimizerGroup.accumulate_metrics`` from
  ``forward_backward``'s ``forward_kwargs``. Typically a
  ``List[List[float]]`` whose outer length equals the total number of
  sequences across all microbatches.

Backend / packing coverage
--------------------------
* Transformers (padded)            \u2713
* Transformers (padding_free, SP)  \u2713
* Megatron (fixed seq len)         \u2713
* Megatron (padding_free, VSL)     \u2713
* DP gather via ``process_group``  \u2713 (propagated by ``add_metric``)
* PP > 1                            \u2713 (non-last stages accumulate 0 tokens
  and are gathered only through the DP group, whose ranks all sit on
  the same PP stage)

The implementation deliberately avoids ``pad_and_stack_tensors`` on
per-microbatch lists: its default ``pad_value=-200`` would defeat the
``labels != -100`` mask under variable-length packing. Instead we
iterate the microbatches and reduce each one independently.
"""
import math
from typing import Any, Dict, List, Optional, Union

from twinkle.data_format import InputFeature, ModelOutput

from .base import Metric


def _align_logps_to_mask(
    ragged: Any,
    mask: 'torch.Tensor',  # noqa: F821
    dtype: 'torch.dtype',  # noqa: F821
) -> Optional['torch.Tensor']:  # noqa: F821
    """Scatter an ``old_logps`` payload into ``[B, T]`` aligned with ``mask``.

    Accepted inputs (covering every shape GRPO rollouts produce):

    * already-aligned ``torch.Tensor`` of shape ``[B, T]`` \u2014 returned
      unchanged (after device/dtype cast).
    * ``List[List[float]]`` (one logp list per sample, response tokens
      only) \u2014 row-wise scattered onto the True positions of ``mask``.
    * ``List[float]`` (one scalar per sample) \u2014 broadcast onto every
      True position within that sample's row.

    Returns ``None`` if the payload cannot be interpreted, signalling
    the caller to silently drop old-policy statistics for this
    microbatch rather than crash training.
    """
    import torch

    device = mask.device
    batch_size, seq_len = mask.shape

    if isinstance(ragged, torch.Tensor):
        t = ragged.to(device=device, dtype=dtype)
        if t.shape == (batch_size, seq_len):
            return t
        # Fall through to the list path (row-wise scatter).
        ragged = [t[i] for i in range(min(batch_size, t.shape[0]))]

    if not isinstance(ragged, (list, tuple)):
        return None

    result = torch.zeros((batch_size, seq_len), dtype=dtype, device=device)
    for i, sample in enumerate(ragged):
        if i >= batch_size:
            break
        pos = mask[i].nonzero(as_tuple=True)[0]
        if len(pos) == 0:
            continue
        if isinstance(sample, (int, float)):
            result[i, pos] = float(sample)
            continue
        vals = torch.as_tensor(sample, dtype=dtype, device=device).flatten()
        n = min(len(pos), int(vals.numel()))
        if n > 0:
            result[i, pos[:n]] = vals[:n]
    return result


class GRPOMetric(Metric):
    """Per-token log-prob diagnostics for GRPO-style RL training.

    Accumulated over every micro-batch in a grad-accum window; aggregated
    (across DP) at :meth:`calculate` time. All statistics are restricted
    to generated tokens (``labels != ignore_index``).

    Fields written to the metric dict (values are formatted strings to
    match the surrounding metric conventions):

    * ``train/policy_confidence`` \u2014 ``exp(mean_new_logp)``, the geometric
      mean of the post-step policy probability assigned to the sampled
      tokens **at the sampler's temperature**. Dropping toward 1.0
      signals entropy / policy collapse. See ``temperature`` arg below.
    * ``train/mean_new_logp`` \u2014 raw mean new-policy log-prob.
    * ``train/logp_std`` \u2014 std of new log-prob across tokens (proxy for
      token-level spread; not true entropy, which would require logits).
    * ``train/mean_old_logp`` \u2014 mean old (sampling) policy log-prob
      (only when ``old_logps`` is supplied).
    * ``train/logp_diff_mean`` \u2014 ``mean(new - old)``. Proportional to
      on-policy update magnitude; invariant under temperature because
      both sides share the same scaling.
    * ``train/approx_kl`` \u2014 Schulman K3 unbiased estimator of
      ``KL(old || new)`` with samples from old:
      ``mean(exp(new - old) - (new - old) - 1)``. Always \u2265 0,
      low-variance, cheap.

    Args:
        device_mesh: DP device mesh (used by :meth:`gather_results`).
        process_group: DP process group.
        ignore_index: Label value marking tokens to exclude from
            statistics (default: ``-100``).
        temperature: Sampler temperature. Both backends divide logits
            by temperature **before** computing ``logps``, so
            ``outputs['logps']`` are log-probabilities of the tempered
            distribution. When ``temperature != 1.0`` the raw
            ``mean_new_logp`` / ``policy_confidence`` therefore measure
            the tempered distribution. Set this to the sampler's
            temperature to recover ``T=1`` values (leave at ``1.0`` to
            report the tempered numbers unchanged, matching the loss's
            own view).
    """

    def __init__(
        self,
        device_mesh=None,
        process_group=None,
        ignore_index: int = -100,
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(device_mesh, process_group, **kwargs)
        self.ignore_index = ignore_index
        self.temperature = float(temperature)
        self.reset()

    def reset(self):
        self.sum_new: float = 0.0
        self.sum_new_sq: float = 0.0
        self.sum_old: float = 0.0
        self.sum_diff: float = 0.0
        self.sum_approx_kl: float = 0.0
        self.n_tokens: int = 0
        self.has_old: bool = False

    @staticmethod
    def _as_mb_list(logps_val) -> Optional[List]:
        """Normalise ``outputs['logps']`` to a list of per-mb 2D tensors."""
        import torch
        if logps_val is None:
            return None
        if isinstance(logps_val, list):
            return logps_val or None
        if torch.is_tensor(logps_val):
            if logps_val.numel() == 0:
                return None
            return [logps_val]
        return None

    def _accumulate_mb(
        self,
        labels: 'torch.Tensor',
        logps: 'torch.Tensor',
        old_slice: Any,
    ) -> int:
        """Reduce one microbatch into ``self.sum_*`` counters.

        Returns ``labels.shape[0]`` so the caller can advance the
        ``old_logps`` slicing cursor even when the microbatch had zero
        generated tokens (e.g. fully-masked prompt-only batch).
        """
        import torch

        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        if not torch.is_tensor(logps) or logps.numel() == 0:
            return labels.shape[0]
        if labels.device != logps.device:
            labels = labels.to(logps.device)

        # Safety-align seq_len (SP / packed edge cases may leave a
        # small off-by-one between labels and logps within a mb).
        if logps.shape[-1] != labels.shape[-1]:
            m = min(logps.shape[-1], labels.shape[-1])
            logps = logps[..., :m]
            labels = labels[..., :m]
        # Safety-align num_seq (mb-local; normally matches exactly).
        if logps.shape[0] != labels.shape[0]:
            n = min(logps.shape[0], labels.shape[0])
            logps = logps[:n]
            labels = labels[:n]

        mask = (labels != self.ignore_index)
        n_tok = int(mask.sum().item())
        num_seq = labels.shape[0]
        if n_tok == 0:
            return num_seq

        # Recover T=1 log-probs if user told us the sampler temperature.
        # At T=1 this is a no-op (temperature field defaults to 1.0).
        # Rescaling keeps ``logp_diff`` / ``approx_kl`` unchanged because
        # both new and old logps receive the same multiplier.
        scale = self.temperature
        logps_f = logps.float()
        if scale != 1.0:
            logps_f = logps_f * scale
        mask_f = mask.float()

        self.n_tokens += n_tok
        self.sum_new += float((logps_f * mask_f).sum().item())
        self.sum_new_sq += float(((logps_f ** 2) * mask_f).sum().item())

        if old_slice is None:
            return num_seq

        aligned = _align_logps_to_mask(old_slice, mask, logps_f.dtype)
        if aligned is None:
            return num_seq
        old_f = aligned.float()
        if scale != 1.0:
            old_f = old_f * scale

        d = logps_f - old_f  # new - old
        self.sum_old += float((old_f * mask_f).sum().item())
        self.sum_diff += float((d * mask_f).sum().item())
        # Schulman K3 estimator of KL(old || new):
        #   samples x ~ old,  r(x) = new(x) / old(x),
        #   k3 = r - 1 - log(r) = exp(new - old) - (new - old) - 1.
        kl = torch.exp(d) - d - 1.0
        self.sum_approx_kl += float((kl * mask_f).sum().item())
        self.has_old = True
        return num_seq

    def accumulate(
        self,
        inputs: Union[InputFeature, List[InputFeature]],
        outputs: ModelOutput,
        *,
        old_logps: Any = None,
        **kwargs,
    ):
        import torch
        if outputs is None:
            return
        logps_val = outputs.get('logps')
        logps_list = self._as_mb_list(logps_val)
        if logps_list is None:
            return

        inputs_list = inputs if isinstance(inputs, list) else [inputs]

        # Megatron ``variable_seq_lengths=False`` cat's all microbatches'
        # logps into one tensor along dim=0 while ``inputs`` remains a
        # per-mb list (all sharing the same fixed seq_len). Merge the
        # per-mb label tensors into a single one so shapes line up and
        # the loop runs exactly once over the whole macro-batch.
        if (torch.is_tensor(logps_val) and len(inputs_list) > 1
                and all(isinstance(i, dict) and i.get('labels') is not None
                        for i in inputs_list)):
            label_tensors = [torch.as_tensor(i['labels']) for i in inputs_list]
            seq_lens = {t.shape[-1] for t in label_tensors}
            if len(seq_lens) == 1:
                merged = torch.cat(label_tensors, dim=0)
                inputs_list = [{'labels': merged}]
        # ``old_logps`` comes in flat over the whole macro-batch; slice
        # it per-mb by cumulative sequence count so we never rely on
        # padded stacking.
        flat_old: Optional[List] = None
        if old_logps is not None and isinstance(old_logps, (list, tuple)):
            flat_old = list(old_logps)

        cursor = 0
        n_mb = min(len(inputs_list), len(logps_list))
        for mb_idx in range(n_mb):
            mb_input = inputs_list[mb_idx]
            if not isinstance(mb_input, dict):
                continue
            labels = mb_input.get('labels')
            if labels is None:
                continue
            import torch
            labels = torch.as_tensor(labels)

            logps_mb = logps_list[mb_idx]

            if flat_old is not None:
                num_seq_est = (labels.shape[0] if labels.dim() >= 2 else 1)
                old_slice = flat_old[cursor:cursor + num_seq_est]
            elif old_logps is not None and hasattr(old_logps, 'shape'):
                # Uncommon: aligned global tensor. Only honour when it
                # exactly matches the single-mb shape; otherwise drop.
                import torch as _torch  # noqa: F811
                old_slice = old_logps if (_torch.is_tensor(old_logps) and old_logps.shape
                                          == logps_mb.shape) else None
            else:
                old_slice = None

            advanced = self._accumulate_mb(labels, logps_mb, old_slice)
            cursor += advanced

    def calculate(self) -> Dict[str, Any]:
        local = [{
            'sum_new': self.sum_new,
            'sum_new_sq': self.sum_new_sq,
            'sum_old': self.sum_old,
            'sum_diff': self.sum_diff,
            'sum_kl': self.sum_approx_kl,
            'n': self.n_tokens,
            'has_old': self.has_old,
        }]
        all_results = self.gather_results(local)

        n_total = sum(r['n'] for r in all_results)
        if n_total == 0:
            self.reset()
            return {}

        sum_new = sum(r['sum_new'] for r in all_results)
        sum_new_sq = sum(r['sum_new_sq'] for r in all_results)
        mean_new = sum_new / n_total
        var_new = max(0.0, sum_new_sq / n_total - mean_new * mean_new)

        results: Dict[str, Any] = {
            'train/policy_confidence': f'{math.exp(mean_new):.4f}',
            'train/mean_new_logp': f'{mean_new:.4f}',
            'train/logp_std': f'{math.sqrt(var_new):.4f}',
        }
        if any(r['has_old'] for r in all_results):
            mean_old = sum(r['sum_old'] for r in all_results) / n_total
            mean_diff = sum(r['sum_diff'] for r in all_results) / n_total
            mean_kl = sum(r['sum_kl'] for r in all_results) / n_total
            results['train/mean_old_logp'] = f'{mean_old:.4f}'
            results['train/logp_diff_mean'] = f'{mean_diff:+.4f}'
            results['train/approx_kl'] = f'{mean_kl:.6f}'

        self.reset()
        return results
