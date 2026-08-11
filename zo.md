check this paper - [https://arxiv.org/html/2510.18228v1](https://arxiv.org/html/2510.18228v1) and this paper - [https://arxiv.org/pdf/2506.09034](https://arxiv.org/pdf/2506.09034)
I checked both. They are closely related: both try to make **zeroth-order (ZO) fine-tuning of LLMs fast enough to be practical while retaining its inference-level memory advantage**, but they attack the inefficiency from different directions. P-GAP appeared in October 2025, while FZOO appeared earlier in June 2025. ([arXiv][1])

### The core distinction

|                       | **P-GAP** — arXiv:2510.18228                                                                             | **FZOO** — arXiv:2506.09034                                                     |
| --------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Main problem attacked | High **variance / bad perturbation directions**                                                          | Too many optimization steps + slow multiple queries                             |
| Main idea             | Restrict perturbations to an estimated low-rank gradient subspace and align them with gradient direction | Make ZO behave roughly like **normalized SGD**, plus batch perturbation queries |
| Perturbation          | Low-dimensional, projected Gaussian                                                                      | Rademacher ±1                                                                   |
| Gradient estimator    | Two-sided                                                                                                | One-sided, multiple directions                                                  |
| Extra mechanism       | Approximate gradient + SVD every (k) steps                                                               | Divide update by std. dev. of perturbed losses                                  |
| Engineering           | Low-rank perturbation                                                                                    | CUDA batching/sign-flip optimization                                            |
| Main tradeoff         | Slight extra memory/computation for much lower estimator variance                                        | More forward work per step, but dramatically fewer steps                        |

P-GAP's key idea is particularly interesting. Every 100 iterations it uses **10 ZO probe perturbations** to build an approximate gradient matrix, performs a low-rank SVD, and keeps ranks such as 128/256/512 for full fine-tuning. For the next 100 iterations it generates perturbations inside that low-dimensional space and projects them so they have controlled alignment with the estimated gradient. ([arXiv][2])

Conceptually:

[
\text{random ZO direction}
\quad\longrightarrow\quad
\text{estimated gradient subspace}
\quad\longrightarrow\quad
\text{gradient-aligned direction}.
]

That attacks one of the fundamental problems with MeZO-like methods: in a model with billions of dimensions, most random directions are terrible directions in which to query the loss. P-GAP explicitly tries to spend its queries in a much smaller, more informative space. ([arXiv][2])

FZOO takes almost the opposite approach. It samples (N) Rademacher directions, evaluates one-sided perturbations,

[
g_t=\frac{1}{\epsilon N}
\sum_i\left[L(\theta+\epsilon u_i)-L(\theta)\right]u_i,
]

and then updates approximately as

[
\theta_{t+1}
============

\theta_t-\eta\frac{g_t}{\sigma_t},
]

where (\sigma_t) is the standard deviation of the perturbed losses. Their analysis connects this scalar normalization to normalized-SGD. ([arXiv][3])

That distinction matters: **FZOO is not really reproducing Adam's coordinate-wise adaptation.** Its theoretical connection is to normalized-SGD, where the entire gradient gets rescaled by essentially one scalar. So I would interpret “Adam-scale speed” in the title as a convergence-speed claim, rather than saying the optimization dynamics are Adam-like. ([arXiv][3])

### What the results actually say

P-GAP has some strong numbers. On OPT-2.7B, for example, it reports SQuAD performance of 74.9 versus 66.8 for MeZO, and on COPA 82.0 versus 76.0. It also improves several OPT-6.7B/13B results. ([arXiv][2])

The speed numbers are more important. On SQuAD/OPT-2.7B, the paper reports P-GAP needing **15.6% of MeZO's iterations and 27.3% of its GPU-hours**, although memory goes from 9.4 GB for MeZO to 11.3 GB for P-GAP. ([arXiv][2])

There is an important nuance there: ordinary gradient-based FT and LoRA in that same table still finish substantially faster—16.8% and 11.5% of the MeZO GPU-hour budget respectively—but require 73.5 GB and 58.5 GB, compared with 11.3 GB for P-GAP. ([arXiv][2])

So I would characterize P-GAP as:

**“Much faster ZO while keeping near-inference memory,”** rather than “faster than conventional fine-tuning.”

FZOO makes an even more aggressive convergence claim. Across its experiments it reports roughly +3 percentage points versus MeZO on average with about 3× fewer forward passes; for RoBERTa-large it reports an 18× reduction in forward passes. It also evaluates models through OPT-66B. ([arXiv][3])

But there is a subtle point in the FZOO results that I think is very important.

**A single FZOO step is actually slower than a MeZO or Adam step.** Their measured A100 timings are:

| Model         |    Adam |    MeZO |    FZOO |
| ------------- | ------: | ------: | ------: |
| OPT-125M      | 0.198 s | 0.137 s | 0.384 s |
| RoBERTa-large | 0.393 s | 0.431 s | 0.605 s |
| OPT-1.3B      | 0.581 s | 0.722 s | 1.663 s |

These are measurements from their own appendix. ([arXiv][3])

So FZOO wins because it claims to require **far fewer optimization steps**, not because each optimization step itself is cheap. Its (N=8) configuration effectively bundles many perturbation evaluations into each step and uses CUDA parallelism to make those evaluations cheaper than doing them sequentially. ([arXiv][3])

That makes the phrase “towards Adam-scale speed” defensible, but I'd want **wall-clock-to-target-accuracy** comparisons, rather than just forward-pass counts, before concluding that it is genuinely Adam-speed in a deployment setting.

### One issue I noticed with P-GAP

This is probably the biggest thing I'd flag in reviewing the two papers together.

**P-GAP does not compare against FZOO.**

P-GAP was submitted on 21 October 2025. It calls MeZO, HiZOO, SubZero and Sparse-MeZO its representative state-of-the-art ZO baselines. ([arXiv][1])

FZOO had already appeared in June 2025, several months earlier, and specifically targets exactly the same problem: accelerating ZO fine-tuning. ([arXiv][4])

I also found no FZOO reference in the P-GAP paper. ([arXiv][2])

That doesn't invalidate P-GAP—the two methods are genuinely different—but it makes the experimental claim of beating representative ZO baselines **incomplete**. A direct P-GAP-vs-FZOO experiment is the obvious missing baseline.

Interestingly, there is some overlap in their reported OPT-13B results. P-GAP reports 92.7/73.8/66.3/85.0 on SST-2/RTE/WSC/SQuAD, whereas FZOO reports 93.7/71.1/63.5/84.8. ([arXiv][2])

That superficially favors P-GAP on three of four tasks and FZOO on SST-2, but **I would not treat that as an official head-to-head comparison** because the implementations, hyperparameter searches and other experimental details aren't guaranteed to be identical.

### The really interesting research direction: combine them

The papers are more complementary than competitive.

A natural hybrid would be roughly:

[
\boxed{
\text{P-GAP subspace}
+
\text{FZOO batched estimator}
+
\text{FZOO normalization}
}
]

P-GAP answers:

> **Where should I perturb?**

FZOO answers:

> **How should I aggregate those queries and choose the effective step size?**

One possible algorithm would estimate (U_r,S_r,V_r) periodically as P-GAP does, sample several (r\times r) perturbations (Z_1,\dots,Z_N), project/alignment-correct them, map them back with

[
\Delta W_i=U_r Z_i V_r^\top,
]

evaluate all (N) perturbations, form the ZO gradient inside that subspace, and normalize the update using FZOO's loss-standard-deviation mechanism. This combination is an inference from the two methods rather than something demonstrated by either paper. ([arXiv][2])

There is one engineering obstacle: **FZOO's fastest CUDA trick depends heavily on perturbations being ±1 Rademacher vectors.** P-GAP's mapped perturbation (U_r ZV_r^\top) is a dense real-valued matrix, so you don't automatically get FZOO's cheap sign-flip implementation. ([arXiv][3])

However, you could potentially use Rademacher perturbations **inside the (r\times r) coefficient space** and then exploit low-rank computation when applying (UZV^\top). That's the experiment I'd be most interested in.

### My assessment

**P-GAP has the more interesting optimization idea.** Reducing the effective perturbation dimension and steering queries toward an approximate gradient subspace directly attacks the fundamental high-dimensional variance problem of ZO. Its results also suggest the estimated subspace can remain useful for ~100 updates, which is important practically. ([arXiv][2])

**FZOO has the cleaner systems/optimizer story.** The normalized-SGD interpretation is simple, the one-sided queries reduce evaluation cost, Rademacher noise enables useful implementation tricks, and it demonstrates scaling to OPT-66B. ([arXiv][3])

If the goal is to push ZO fine-tuning forward, though, I wouldn't choose one and discard the other. **The strongest next experiment is P-GAP's low-dimensional directional variance reduction combined with FZOO's batch/statistical normalization**, benchmarked under identical hardware and data.

And I would evaluate the four methods

[
\text{MeZO},\quad
\text{FZOO},\quad
\text{P-GAP},\quad
\text{P-GAP+FZOO}
]

using **wall-clock time to a fixed target validation score, peak memory, total forward-equivalent FLOPs, and final task performance**. That would answer the question neither paper currently answers: *what is actually the best ZO recipe when algorithmic variance reduction and systems-level acceleration are used together?*

[1]: https://arxiv.org/abs/2510.18228 "[2510.18228] Towards Fast LLM Fine-tuning through Zeroth-Order Optimization with Projected Gradient-Aligned Perturbations"
[2]: https://arxiv.org/html/2510.18228v1 "Towards Fast LLM Fine-tuning through Zeroth-Order Optimization with Projected Gradient-Aligned Perturbations"
[3]: https://arxiv.org/pdf/2506.09034 "FZOO: Fast Zeroth-Order Optimizer for Fine-Tuning Large Language Models towards Adam-Scale Speed"
[4]: https://arxiv.org/abs/2506.09034?utm_source=chatgpt.com "FZOO: Fast Zeroth-Order Optimizer for Fine-Tuning Large Language Models towards Adam-Scale Speed"


Okay so is this like a potential paper and also did anyone do this before?
Yes — **there is potentially a paper here**, but I would **not** frame the contribution as simply “we combine P-GAP and FZOO.” The 2026 literature has moved enough that reviewers could call that an incremental combination.

As of **August 11, 2026**, I did **not** find a paper implementing the exact combination:

[
\boxed{
\text{P-GAP gradient-aligned subspace}
+
\text{FZOO multi-query estimator}
+
\text{FZOO loss-std normalization}
+
\text{efficient batched execution}
}
]

So the exact recipe appears open based on the literature I could find. But there are some **very close neighbors**.

### The biggest prior-art risks

**ZO-Muon (Feb. 2026)** is probably the most important one. It explicitly combines a **low-dimensional projected/subspace ZO estimator** with a more sophisticated optimizer transformation, and it already uses **multiple queries per step**. For LLMs, the authors search (N_q\in{4,8,16}) and often use (N_q=4). They explicitly discuss FZOO and characterize it as a multi-query method that uses function-evaluation variance for normalization. ([arXiv][1])

So a reviewer could look at:

> P-GAP subspace + FZOO optimizer

and say:

> “ZO-Muon already demonstrated subspace ZO + optimizer-level acceleration. What is fundamentally new?”

That's the main issue.

Then there is **ZO-Act**, posted July 1, 2026. This one gets even closer structurally. It defines an activation-derived low-dimensional coefficient space, draws (q) independent perturbations **inside that space**, averages the resulting forward-difference estimates, and then updates the coefficient matrix with Adam. ([arXiv][2])

In other words:

[
\text{low-dimensional subspace}
+
\text{multi-query ZO}
+
\text{adaptive optimizer}
]

has definitely been done.

There is also **BSZO**, which uses several perturbation directions within a subspace and combines their measurements with a Bayesian/Kalman estimator, including adaptive handling of noise. ([arXiv][3])

And **LOZO+**, accepted at ICML 2026, already attacks the problem of selecting better low-rank subspaces using loss feedback rather than relying on random subspaces. ([OpenReview][4])

So the area is pretty active.

---

## But I think there is a better paper hiding here

Instead of:

> **“We combine P-GAP and FZOO.”**

I'd turn the research question into:

> **How should multi-query ZO optimization be performed inside an informative gradient-aligned low-dimensional subspace?**

That's substantially more interesting.

There's actually a theoretical reason for doing this. The **Multi-Query Paradox** paper from September 2025 showed that simply averaging more random ZO queries is not necessarily query-efficient. Under a fixed query budget, their theory says ordinary ZO averaging can actually favor **one query per iteration**, whereas their geometry-aware Projection Alignment estimator benefits from multiple queries. ([arXiv][5])

That immediately creates a question that neither P-GAP nor FZOO really answers:

[
\boxed{\text{What happens to multi-query efficiency once queries are gradient-aligned?}}
]

That's much stronger.

### Here's the particular idea I'd investigate

P-GAP makes its directions informative by aligning them with an estimated gradient. But suppose you generate (N=8) P-GAP directions.

If all eight directions are strongly aligned with approximately the same gradient,

[
u_1,\ldots,u_8 \approx \hat g,
]

then they're highly correlated.

You might be paying for eight forward passes but getting much less than eight independent pieces of information.

And FZOO's normalization depends on the standard deviation of the losses

[
\sigma_L=
\operatorname{Std}\left[
L(\theta+\epsilon u_1),
\dots,
L(\theta+\epsilon u_N)
\right].
]

Strongly aligning all the (u_i)'s could change that statistic dramatically.

**Neither paper studies this interaction.**

That is a real research question.

---

# A potentially novel algorithm

Instead of simply sampling eight P-GAP directions, construct them as

[
u_i =
\rho,\hat g
+
\sqrt{1-\rho^2},v_i,
]

where

[
v_i^\top \hat g =0
]

and ideally

[
v_i^\top v_j\approx0.
]

Now every query has:

**gradient alignment**

[
\cos(u_i,\hat g)\approx\rho
]

while also having **query diversity**

[
u_i^\top u_j
]

kept small.

So you'd explicitly balance:

[
\boxed{\text{alignment vs diversity}}
]

rather than maximizing alignment alone.

And all of this happens inside P-GAP's low-dimensional space:

[
u_i = U_r C_i V_r^\top.
]

Then estimate the gradient using the (N) directional evaluations.

You could compare three aggregation rules:

[
\text{simple average}
]

vs.

[
\text{FZOO normalized average}
]

vs.

[
\text{projection-aware aggregation}.
]

That would connect P-GAP, FZOO and the Multi-Query Paradox literature rather than merely gluing two algorithms together. ([arXiv][5])

---

# An even stronger angle: subspace-aware normalization

FZOO uses

[
\frac{\hat g}{\sigma_L}.
]

But FZOO derives this in its normal perturbation setting and relates it to normalized SGD. ([arXiv][6])

Once you restrict the perturbations from the full dimension (d) to an effective dimension (r),

[
d \gg r,
]

the distribution of directional derivatives changes.

And once P-GAP biases those directions toward (\hat g), it changes again.

So perhaps the appropriate normalization isn't simply

[
\frac{1}{\sigma_L}.
]

It might be something like

[
\frac{1}
{\sigma_L,c(r,N,\rho)}
]

where

* (r) = subspace rank,
* (N) = queries per iteration,
* (\rho) = alignment strength.

If you could **derive this theoretically**, that would make the paper much stronger.

Then your argument isn't:

> “We put FZOO inside P-GAP.”

It's:

> **“Existing multi-query normalization is derived for isotropic ZO perturbations. We show that gradient-aligned low-rank perturbations alter the loss-statistic geometry, derive an appropriate subspace-aware normalization, and design diversity-preserving queries.”**

That's a legitimate optimization contribution.

---

## And there's a systems problem too

FZOO's implementation advantage comes partly from Rademacher (\pm1) perturbations and efficient parallel execution. ([arXiv][7])

P-GAP perturbations of the form

[
U_r C V_r^\top
]

are generally dense real-valued perturbations.

So naïvely combining the two **loses some of FZOO's systems advantage**.

That's another potential contribution:

[
C_i\in{-1,+1}^{r\times r}
]

and perform the cheap random perturbation **in coefficient space**, then map it through (U_r,V_r).

Instead of materializing

[
U_rC_iV_r^\top
]

for every query, write a fused implementation that evaluates the low-rank perturbation directly.

This systems aspect is increasingly relevant because a May 2026 paper showed that ZO fine-tuning can be reorganized as an inference-style workload and reported large runtime improvements simply from changing the execution system. ([arXiv][8])

So **wall-clock efficiency** needs to be part of the contribution now.

---

# Where I'd rate the ideas

| Idea                                              | Novelty potential                            |
| ------------------------------------------------- | -------------------------------------------- |
| Run FZOO on P-GAP                                 | 🟡 Weak–moderate                             |
| P-GAP + FZOO + CUDA batching                      | 🟡 Moderate                                  |
| Low-rank FZOO with theoretical convergence        | 🟡 Moderate                                  |
| Subspace-aware FZOO normalization                 | 🟢 Good                                      |
| Alignment + multi-query diversity                 | 🟢 **Very good**                             |
| Alignment/diversity + rank-aware normalization    | 🟢🟢 **Strong**                              |
| Above + efficient low-rank batched implementation | 🟢🟢 **Potentially strong systems/ML paper** |

The last two are where I'd focus.

### The paper could have a very clean story

Something like:

> **Aligned Multi-Query Zeroth-Order Optimization for LLM Fine-Tuning**

with the observation:

[
\text{P-GAP: better query directions}
]

[
+
]

[
\text{FZOO: multiple queries + normalization}
]

but:

[
\boxed{
\text{aligned queries become correlated}
}
]

so naïvely combining them wastes queries.

Then your contribution is a method that maintains

[
\underbrace{\text{gradient alignment}}*{\text{good direction}}
+
\underbrace{\text{query diversity}}*{\text{good information}}
+
\underbrace{\text{subspace-aware normalization}}_{\text{good step size}}.
]

That is a much more defensible novelty claim.

And experimentally I'd make the critical ablation:

[
\begin{array}{l}
\text{MeZO}\
\text{FZOO}\
\text{P-GAP}\
\text{P-GAP + naïve FZOO}\
\text{P-GAP + diverse queries}\
\text{P-GAP + diverse queries + normalization}
\end{array}
]

plus ZO-Muon, ZO-Act, AGZO, LOZO+, and BSZO because those are now important contemporary baselines. ([arXiv][1])

**So yes: I think there is a paper opportunity. But the genuinely interesting paper isn't “P-GAP + FZOO”; it's the unexplored interaction between *gradient alignment, query diversity, effective subspace dimension, and FZOO-style normalization*.**

If you want, I can next turn this into a **concrete proposed algorithm with equations/pseudocode and a paper-worthy novelty claim**, and we can see whether it survives a reviewer-style novelty test.

[1]: https://arxiv.org/html/2602.17155v1?utm_source=chatgpt.com "Powering Up Zeroth-Order Training via Subspace Gradient Orthogonalization"
[2]: https://arxiv.org/pdf/2607.01125v1?utm_source=chatgpt.com "ZO-Act: Efficient Zeroth-Order Fine-Tuning via One-Shot Activation-Informed Low-Rank Subspaces"
[3]: https://arxiv.org/abs/2601.01452?utm_source=chatgpt.com "[2601.01452] Robust and Efficient Zeroth-Order LLM Fine-Tuning via Adaptive Bayesian Subspace Optimizer"
[4]: https://openreview.net/forum?id=QNScXX6osL&utm_source=chatgpt.com "LOZO+: Provably Efficient Zeroth-Order Fine-Tuning via Greedy Low-Rank Subspace Selection | OpenReview"
[5]: https://arxiv.org/html/2509.15552v1?utm_source=chatgpt.com "The Multi-Query Paradox in Zeroth-Order Optimization"
[6]: https://arxiv.org/abs/2506.09034v2?utm_source=chatgpt.com "[2506.09034v2] FZOO: Fast Zeroth-Order Optimizer for Fine-Tuning Large Language Models towards Adam-Scale Speed"
[7]: https://arxiv.org/pdf/2506.09034v1?utm_source=chatgpt.com "FZOO: Fast Zeroth-Order Optimizer for Fine-Tuning Large Language Models towards Adam-Scale Speed"
[8]: https://arxiv.org/abs/2605.28760?utm_source=chatgpt.com "LLM Zeroth-Order Fine-Tuning is an Inference Workload"


im thinking of doing training on npu using QC HMX infrastructure that is in ggml-hexagon
Yes — **that changes the paper substantially, and in a good way**.

The more compelling paper would not be “P-GAP + FZOO.” It becomes:

> **Forward-only LLM fine-tuning on a mobile NPU, using hardware-aware zeroth-order optimization on Qualcomm Hexagon HMX.**

That is a much clearer systems + ML contribution.

The fit is unusually natural because the current Snapdragon `ggml-hexagon` backend is designed around forward inference on the Hexagon Tensor Processor, and current llama.cpp contains dedicated HMX matmul and flash-attention paths plus FP16 HMX tile helpers. The backend is still marked experimental, but it already supports NPU offload and multiple HTP devices/sessions. ([GitHub][1]) Qualcomm has also now publicly documented the V81 HMX ISA. ([Qualcomm Docs][2])

## Why ZO makes sense specifically for HMX

Normal training wants:

[
\text{forward}+\text{backward}+\text{optimizer}
]

and getting a complete backward framework running efficiently on a mobile NPU is a huge undertaking.

ZO instead needs essentially:

[
L(\theta+\epsilon z)
]

and perhaps

[
L(\theta-\epsilon z)
]

followed by a tiny amount of scalar/vector update logic.

So HMX can remain almost entirely an **inference engine**.

Conceptually:

[
\boxed{\text{NPU inference runtime}}
\quad+\quad
\boxed{\text{ZO optimizer}}
\quad\Rightarrow\quad
\boxed{\text{NPU training}}
]

No attention backward kernel. No GEMM backward. No activation-gradient storage. No autograd engine.

That is a strong story by itself.

---

## But here's the important implementation problem

You **do not want to implement ZO naïvely** as:

```text
copy W
add perturbation to every W
repack W
run NPU
restore W
repeat N times
```

That could destroy the entire advantage.

This matters especially with quantized GGUF models because the Hexagon backend uses backend-specific repacked representations for some quantized formats; current documentation explicitly discusses repacking buffers for Q4_0, Q8_0 and MXFP4. ([GitHub][3]) Current Hexagon support also includes F16 alongside several quantized formats. ([GitHub][4])

So I would make **“perturbation without weight materialization”** one of the central systems ideas.

### Example

Suppose P-GAP gives you a low-rank perturbation

[
\Delta W_i=U C_i V^\top.
]

Instead of constructing

[
W_i=W+\epsilon UC_iV^\top
]

and feeding that new weight to HMX, exploit linearity:

[
x(W+\epsilon UC_iV^\top)
========================

xW+
\epsilon(xU)C_iV^\top.
]

Now the base model:

[
W
]

stays exactly where it is.

For a Q4 base model it can remain:

* quantized,
* HMX packed,
* resident/reused,

while the perturbation is evaluated as a few **small FP16 HMX GEMMs**.

That is beautiful for this architecture.

You'd basically have an HMX operator such as

```text
MUL_MAT_ZO(
    base_quantized_weight,
    activation,
    U,
    C_i,
    V,
    epsilon
)
```

which computes

[
Wx+\epsilon \Delta W_i x
]

without ever generating (W+\epsilon\Delta W_i).

---

# And FZOO becomes especially interesting

Suppose FZOO wants (N=8) perturbation queries.

Rather than launching them one at a time:

[
z_1,z_2,\ldots,z_8,
]

you could expose the **query dimension as another batching dimension**.

Something conceptually like

[
X\in
\mathbb R^{N_q\times B\times S\times d}.
]

Then HMX evaluates multiple perturbed forward passes together.

The accelerator likes large matrix operations, so:

[
\boxed{\text{multi-query ZO}}
]

isn't merely an optimization algorithm choice anymore.

It becomes a way of **increasing NPU utilization**.

That is exactly the type of algorithm/hardware co-design story reviewers like.

---

## I would actually call the project something like

**NPU-ZO: Hardware-Aware Zeroth-Order LLM Fine-Tuning on Mobile NPUs**

and your architecture could look like:

```text
                    ARM CPU
             ┌──────────────────┐
             │ ZO control plane │
             │                  │
             │ loss statistics  │
             │ query seeds      │
             │ normalization    │
             │ P-GAP subspace   │
             │ parameter update │
             └────────┬─────────┘
                      │
                      ▼
             Qualcomm Hexagon
        ┌──────────────────────────┐
        │        HMX NPU           │
        │                          │
        │ Quantized base model     │
        │       +                  │
        │ low-rank perturbations   │
        │       +                  │
        │ batched ZO queries       │
        │                          │
        │ forward / loss           │
        └──────────────────────────┘
```

The CPU is the **optimizer/control plane**.

HMX is the **forward-compute plane**.

That's a clean division.

---

# There is one bigger problem we need to solve

This is probably the most interesting technical issue.

Perturbing the parameters is easy.

But after calculating the ZO gradient:

[
\hat g_t,
]

you eventually need:

[
W_{t+1}=W_t-\eta\hat g_t.
]

If you're using an FP16 1B model, you could simply modify the weights in unified memory. The Hexagon backend explicitly takes advantage of Snapdragon's unified memory model. ([GitHub][3])

But if your base model is Q4, **modifying/requantizing/repacking billions of weights every optimization step is precisely what we don't want**.

So there are actually two paths.

### Path A — first prototype: FP16

Start with something small like:

[
\text{Llama-3.2-1B / Qwen 0.5–1.5B}
]

using **F16 parameters**.

Then:

1. CPU perturb weights using seeds.
2. HMX forward.
3. Restore.
4. HMX second forward.
5. CPU computes ZO update.
6. Update FP16 weights.
7. Repeat.

The Hexagon backend already supports F16. ([GitHub][4])

This proves:

> **actual LLM training on the NPU without backpropagation.**

I'd do this first.

---

# Path B — potentially the real paper

Keep:

[
W_0
]

frozen and quantized.

Represent the learned change as

[
W=W_0+\Delta W.
]

And keep

[
\Delta W
]

low-rank:

[
\Delta W=AB^\top
]

or in a P-GAP-derived subspace

[
\Delta W=UCV^\top.
]

Now only (C), or (A,B), are trainable.

Therefore the big Q4 weights **never change**.

This gives:

[
\boxed{
\text{frozen Q4 model}
+
\text{FP16 trainable subspace}
}
]

which is practically perfect for a phone NPU.

The HMX computation becomes:

[
xW_0+
xUCV^\top.
]

The first term uses the optimized quantized HMX path.

The second is a tiny FP16 low-rank path.

And ZO operates only on (C):

[
C+\epsilon Z_i.
]

Now your optimization dimension can drop enormously.

For a (4096\times4096) matrix,

[
W:16.8\text{ million parameters}.
]

But if

[
C\in\mathbb R^{64\times64},
]

that's only

[
4096
]

perturbed coefficients.

Now the hardware and optimization arguments reinforce each other:

[
\text{low-dimensional ZO}
\rightarrow
\text{less variance}
]

and

[
\text{low-dimensional ZO}
\rightarrow
\text{cheap HMX perturbation}.
]

---

## There's a novelty caveat

Parameterizing the trainable part in a low-dimensional space is not by itself novel anymore; things like ZO-Act and various ZO-LoRA approaches are close.

But this particular framing is more interesting:

> **Design the ZO parameterization around the native HMX execution path so that a quantized inference model can be fine-tuned without backward kernels, without dequantizing the base model, and without repacking model weights between queries.**

That's much less like an optimizer paper.

It's an **on-device training system**.

---

# What I would target experimentally

Your killer table could eventually be something like:

| System       | Device  | Backward | Base   | Training memory | Energy/update | Time-to-target |
| ------------ | ------- | -------- | ------ | --------------: | ------------: | -------------: |
| PyTorch LoRA | CPU     | Yes      | FP16   |               … |             … |              … |
| CPU MeZO     | CPU     | No       | FP16   |               … |             … |              … |
| CPU FZOO     | CPU     | No       | FP16   |               … |             … |              … |
| NPU naïve ZO | HMX     | No       | FP16   |               … |             … |              … |
| **NPU-ZO**   | **HMX** | **No**   | **Q4** |           **…** |         **…** |          **…** |

Then demonstrate:

[
\boxed{\text{training on a production mobile NPU}}
]

with measurements of:

* wall-clock training time,
* joules/update,
* peak system memory,
* NPU utilization,
* CPU utilization,
* temperature/throttling,
* perturbation-query throughput,
* final task accuracy.

Mobile inference literature is already emphasizing energy gains from Hexagon, so **energy-per-training-step** would be an especially valuable metric rather than reporting only speed. ([arXiv][5])

## What I would do first

I wouldn't start by implementing P-GAP.

I'd build the smallest possible proof:

[
\boxed{\text{MeZO + FP16 + HMX}}
]

on perhaps a 0.5B–1B model.

Get this loop working:

```text
seed z
    ↓
perturb parameters
    ↓
HMX forward
    ↓
loss+
    ↓
same seed, perturb -
    ↓
HMX forward
    ↓
loss-
    ↓
ZO scalar
    ↓
apply update
```

Then benchmark **where the time goes**.

If weight perturbation dominates, that confirms the need for the fused low-rank approach.

Then implement:

[
\boxed{\text{low-rank perturbation-in-HMX}}
]

followed by multi-query batching/FZOO.

And *then* P-GAP/alignment becomes the algorithmic cherry on top.

I think **“training an LLM directly through `ggml-hexagon`/HMX using forward-only ZO” is potentially a much better paper direction than the original P-GAP+FZOO idea alone**. The strongest version would demonstrate that an inference-oriented, quantized NPU can perform useful personalization/fine-tuning **without ever implementing backpropagation or rewriting the quantized base weights**. ([GitHub][6])

[1]: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/hmx-ops.c?utm_source=chatgpt.com "llama.cpp/ggml/src/ggml-hexagon/htp/hmx-ops.c at master · ggml-org/llama.cpp · GitHub"
[2]: https://docs.qualcomm.com/doc/80-N2040-62/80-N2040-62_REV_AA_Qualcomm_Hexagon_V81_HMX_Programmers_Reference_Manual.pdf?utm_source=chatgpt.com "Qualcomm Technologies, Inc."
[3]: https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/developer.md?utm_source=chatgpt.com "llama.cpp/docs/backend/snapdragon/developer.md at master · ggml-org/llama.cpp · GitHub"
[4]: https://github.com/ggml-org/llama.cpp/discussions/22791?utm_source=chatgpt.com "No performance benefit with OpenCL/HTP on Android - same speed as CPU on Snapdragon 8 Gen 3 · ggml-org llama.cpp · Discussion #22791 · GitHub"
[5]: https://arxiv.org/abs/2606.11257?utm_source=chatgpt.com "Energy-Efficient On-Device RAG on a Mobile NPU: System Design and Benchmark on Snapdragon X Elite"
[6]: https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md?utm_source=chatgpt.com "llama.cpp/docs/backend/snapdragon/README.md at master · ggml-org/llama.cpp · GitHub"



so like is this icassp level paper
Yes — **if you can get the NPU training loop actually working and show meaningful gains, this is absolutely ICASSP-level material**. ICASSP 2027 explicitly includes *Applied Signal Processing Systems*, *Machine Learning and Generative AI*, *Green Signal Processing*, *IoT*, and *Signal Processing in Emerging Technologies*, so the topic fits the venue well. ([ICASSP 2027][1])

The important distinction is that the **idea alone is not enough**. An ICASSP paper needs a tight technical contribution plus convincing experiments. For your project, I think the minimum convincing story would be:

1. **First demonstration / practical system for forward-only LLM fine-tuning on Qualcomm HMX through `ggml-hexagon`.**
2. A hardware-aware ZO design that avoids backward kernels and preferably avoids repeatedly rewriting or repacking the full model.
3. Measurements on real Snapdragon hardware showing something useful such as lower memory, lower energy, or faster time-to-target than CPU ZO.
4. At least one real downstream fine-tuning/personalization task showing that the model actually learns.

If you only do:

[
\text{MeZO implementation on HMX}
]

and report that it runs, I'd call that **borderline ICASSP** unless the systems result is genuinely striking.

If you do:

[
\text{MeZO/HMX}
+
\text{optimized perturbation execution}
+
\text{energy/memory benchmarks}
]

that's **solid ICASSP territory**.

And if you achieve:

[
\boxed{
\text{quantized frozen HMX model}
+
\text{low-rank trainable state}
+
\text{batched ZO queries}
+
\text{no backward}
}
]

with, say, a convincing **2–5× wall-clock/energy advantage over CPU ZO** while retaining comparable task accuracy, I'd consider that a **strong ICASSP submission**.

The fact that ICASSP 2027 regular papers are only **4 technical pages plus an optional references page** also favors this kind of focused contribution. You don't need to solve every theoretical aspect of ZO optimization in one submission; you need one clean result and a good ablation. ([ICASSP 2027][2])

### I would scope the ICASSP version narrowly

I would **not** try to cram P-GAP + FZOO + a new optimizer + HMX kernels + quantization theory into the first paper.

A cleaner ICASSP paper would be:

> **Forward-Only LLM Fine-Tuning on Mobile NPUs via Zeroth-Order Optimization**

Core contribution:

[
\text{Inference-only HMX backend}
\rightarrow
\text{actual model adaptation}
]

without implementing backpropagation.

Then compare:

| Method          | Execution | Backprop | Peak RAM | Energy | Time-to-target |
| --------------- | --------- | -------: | -------: | -----: | -------------: |
| CPU MeZO        | CPU       |       No |        — |      — |              — |
| CPU LoRA        | CPU/GPU   |      Yes |        — |      — |              — |
| HMX inference   | NPU       |        — |        — |      — |              — |
| **Your HMX-ZO** | **NPU**   |   **No** |    **—** |  **—** |          **—** |

Then have one important systems optimization, perhaps:

[
\text{naïve weight perturbation}
\quad\text{vs}\quad
\text{on-the-fly low-rank perturbation}.
]

That alone gives you an excellent ablation.

### What would make reviewers care

The strongest headline isn't:

> “ZO works on Snapdragon.”

It is something measurable like:

> **We enable LLM personalization directly on an inference-oriented mobile NPU without backward kernels, reducing training energy by X× and peak memory by Y× compared with CPU-based fine-tuning.**

That's a strong ICASSP-style claim because it combines an algorithmic insight with signal-processing hardware constraints.

One warning: **“first NPU LLM training” is a claim we'd need to search extremely carefully before putting it in the paper.** There are mobile adaptation, accelerator training, PEFT and on-device learning papers, so we'd want a very precise first-of-its-kind statement rather than an overly broad one.

Also, if you're targeting **ICASSP 2027**, the current submission deadline is **September 16, 2026**, so the scope needs to be very focused given today's date of August 11. ([ICASSP 2027][1])

If you can already run inference through HMX, I'd say the **highest-value next step is to prove one trainable parameter block can be updated through a complete ZO loop on-device**. Once that works, you'll know very quickly whether you have an ICASSP paper or whether the bottleneck forces a different design.

[1]: https://2027.ieeeicassp.org/call-for-papers/?utm_source=chatgpt.com "Call for Papers - 2027 IEEE International Conference on Acoustics, Speech, and Signal Processing"
[2]: https://2027.ieeeicassp.org/wp-content/uploads/sites/13/2019/08/ICASSP2027-CallForPapers-July1.pdf?utm_source=chatgpt.com "2027.IEEEICASSP"
