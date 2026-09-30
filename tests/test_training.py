import numpy as np
import torch

from teemoe.train.aggregation import sample_plan
from teemoe.train.common import learning_rate
from teemoe.train.text import plan, response_loss


def test_text_plan_uses_every_example_once():
    lengths = list(np.random.default_rng(0).integers(10, 100, 101))
    batches = plan(lengths, batch=8, bucket=4, seed=3)
    assert sorted(i for b in batches for i in b) == list(range(101))
    assert all(len(b) == 8 for b in batches[:-1]) and len(batches[-1]) == 101 % 8


def test_schedule():
    rates = [learning_rate(u, 100, peak=1.0, warmup_ratio=0.05, final_fraction=0.1) for u in range(100)]
    assert rates[0] == 0.2 and max(rates) == 1.0 and abs(rates[-1] - 0.1) < 1e-9
    assert learning_rate(7, 100, peak=3.0, schedule="constant") == 3.0


def test_chunked_loss_gradient_matches_autograd():
    torch.manual_seed(0)
    head = torch.nn.Linear(8, 30, bias=False).requires_grad_(False)
    student, teacher = torch.randn(300, 8, requires_grad=True), torch.randn(300, 8)
    labels = torch.randint(0, 30, (300,))
    gradient, total = response_loss(student.detach(), teacher, labels, head, 0.5, 0.05, 123.0)
    logp, base = head(student).log_softmax(-1), head(teacher).log_softmax(-1)
    loss = (torch.nn.functional.nll_loss(logp, labels, reduction="sum")
            + 0.5 * (base.exp() * (base - logp)).sum() + 0.05 * (logp.exp() * (logp - base)).sum())
    (loss / 123.0).backward()
    assert torch.allclose(gradient, student.grad, atol=1e-6) and abs(total - loss.item()) < 1e-3


def test_editor_sampler_is_half_uniform_half_balanced():
    groups = np.repeat([0, 1, 2], [1000, 10, 1])
    batches = sample_plan(groups, updates=200, batch=32, seed=1)
    assert all(len(b) == 32 for b in batches)
    share = np.mean([np.isin(b, np.flatnonzero(groups == 2)).mean() for b in batches])
    assert 0.1 < share < 0.25  # the single-window group gets about a sixth of each batch
