# Paper ablations and baselines

These scripts evaluate the comparisons in the paper's tables with the regular
evaluation commands (`teemoe.eval.gift`, `.cik`, `.tse`); everything after the
benchmark name is passed on to them.

| Row | Command |
|---|---|
| Aggregation / native / analysis expert alone | `python paper/ablations.py expert:aggregation cik` (or `expert:native`, `expert:analysis`) |
| Top-1 routing | `python paper/ablations.py top1 cik` |
| Equal-weight adapters | `python paper/ablations.py fixed:0.3333333333,0.3333333333,0.3333333333 cik` |
| Full-strength adapters | `python paper/ablations.py fixed:1,1,1 cik` |
| Qwen3.6-27B without adapters | `python paper/ablations.py base cik` |
| XGBoost + Toto-FnF | `python paper/ablations.py ensemble:reference gift --gpus 0 1 2 3 4 5 6 7` |
| XGBoost-weighted ensemble (8) | `python paper/ablations.py ensemble:router gift …` |
| Toto-FnF | `python paper/ablations.py ensemble:fnf gift …` |
| Equal-weight ensemble (8 / 13) | `python paper/ablations.py ensemble:equal8 gift …` (`ensemble:equal13`) |
| Joint | train with `paper/joint.py` (see its docstring), then `python paper/ablations.py joint cik --checkpoint artifacts/joint/checkpoint` |

Replace `cik` by `gift` or `tse` for the other benchmarks, and give each run its
own `--output`. With fixed weights the controller still decides between numerical
and text output; an expert alone answers with text except the aggregation expert
on GIFT-Eval. Numerical ensembles answer every forecast numerically; on Context
is Key their 25 trajectories are independent draws from the forecast quantiles.

`convert_checkpoint.py` turns a model trained with the paper's reproduction code
into the checkpoint format that `TeeMoE.from_pretrained` loads.
