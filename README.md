# Conformal LLM Routing with Distribution-Free Safety Guarantees

[Paper](https://aclanthology.org/2026.acl-srw.70/) | [PDF](https://aclanthology.org/2026.acl-srw.70.pdf)

This repository contains the code and experiment materials for **Conformal LLM Routing with Distribution-Free Safety Guarantees**, published at ACL 2026 Student Research Workshop.

## Overview

LLM routing sends each query to a cheaper model when it is likely to be sufficient and to a more expensive model otherwise. Most routing methods optimize quality and cost, but do not certify how often the cheap model will be worse than the expensive model among the queries it receives.

This work uses a conformally calibrated input-based gate to control that risk. Given a user-selected violation tolerance $\alpha$ and confidence level $1-\delta$, the router selects a threshold that guarantees:

$$
\Pr\left(\Pr(Y=0 \mid s(X) \ge t^*) > \alpha\right) \le \delta.
$$

Here, $Y=0$ means the cheap model is incorrect while the expensive model is correct. Queries with gate scores above the calibrated threshold are routed to the cheap model; all others are routed to the expensive model.

## Method

1. Embed each input query with `BAAI/bge-base-en-v1.5`.
2. Train a logistic regression gate to predict whether routing to the cheap model is safe.
3. Use a held-out calibration set to choose the lowest routing threshold whose Clopper-Pearson upper confidence bound is at most $\alpha$.
4. Route only queries that meet the calibrated threshold to the cheaper model.

The method makes routing decisions from the input alone. It does not require running the cheap model before deciding whether to escalate the query.

## Experiments

We evaluate routing between:

- Cheap model: Mixtral-8x7B-Instruct
- Expensive model: GPT-4-1106-preview
- Data: RouterBench correctness labels for GSM8K and MMLU
- Gate: Logistic regression over BGE embeddings
- Calibration confidence: $\delta = 0.10$

The two models have a 24.5x difference in mean per-query cost in RouterBench.

## Main Results

| Dataset | Method | Coverage | Violation | Target met? | Cost savings |
| --- | --- | ---: | ---: | :---: | ---: |
| GSM8K ($\alpha=0.30$) | Conformal | 36.7% | 0.280 | Yes | 35% |
| GSM8K ($\alpha=0.30$) | Validation-tuned | 59.6% | 0.317 | No | 58% |
| GSM8K ($\alpha=0.30$) | Always cheap | 100.0% | 0.354 | No | 97% |
| MMLU ($\alpha=0.20$) | Conformal | 90.3% | 0.191 | Yes | 87% |
| MMLU ($\alpha=0.20$) | Validation-tuned | 92.6% | 0.196 | Yes | 90% |
| MMLU ($\alpha=0.20$) | Always cheap | 99.3% | 0.215 | No | 96% |

On GSM8K, validation tuning exceeds the target violation rate despite higher coverage. The conformal router sacrifices some coverage to preserve the requested safety constraint. On MMLU, conformal routing sends 90.3% of queries to the cheaper model while meeting the target and reducing cost by 87%.

## Feasibility Diagnostic

Safe routing depends on both the task and the cheap model. A positive-coverage routing threshold is feasible when the gate can meet the required separation:

$$
\frac{\operatorname{TPR}(t)}{\operatorname{FPR}(t)} \ge
C(\pi,\alpha) = \frac{(1-\pi)(1-\alpha)}{\pi\alpha},
$$

where $\pi$ is the base rate at which it is safe to route to the cheap model. Higher values of $C(\pi,\alpha)$ require stronger gate separation.

## Citation

```bibtex
@inproceedings{uddin-bauer-2026-conformal,
  title = {Conformal {LLM} Routing with Distribution-Free Safety Guarantees},
  author = {Uddin, Iqtedar and Bauer, Andr{\'e}},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop)},
  year = {2026},
  pages = {791--799},
  publisher = {Association for Computational Linguistics},
  url = {https://aclanthology.org/2026.acl-srw.70/},
  doi = {10.18653/v1/2026.acl-srw.70}
}
```
