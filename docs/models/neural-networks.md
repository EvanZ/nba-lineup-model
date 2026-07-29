# Neural Networks

The neural modeling program is intended to measure lineup interactions rather
than replace RAPM with an opaque model immediately. Each architecture must add
one identifiable capability while retaining the same possession sample,
chronological splits, target, and evaluation artifacts.

## Stack

The project uses [PyTorch](https://pytorch.org/docs/stable/index.html) for model
definitions and [Lightning](https://lightning.ai/docs/pytorch/stable/) for
training loops, checkpoints, deterministic seeding, device selection, and
early stopping. CPU is the default accelerator. Apple MPS can be selected as an
experiment, but it is not assumed to be faster for these small models.

Prefect remains responsible for data workflows. Lightning owns the inner model
training loop.

## Modeling ladder

| Stage | Aggregation | New capability | Status |
| --- | --- | --- | --- |
| Additive neural RAPM | Signed sum | Validates tensors, optimization, and artifacts | Implemented |
| Deep Sets | Nonlinear pooled sets | Nonlinear lineup strength without pairwise attention | Planned |
| Transformer | Self-attention | Player-player and context-dependent interactions | Planned |

Deep Sets is an important ablation. If it matches the Transformer, the benefit
comes from nonlinear aggregation rather than attention specifically. This
follows the permutation-invariant construction in
[Deep Sets](https://arxiv.org/abs/1703.06114). The later attention model follows
the encoder mechanism introduced in
[Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## Possession sample

The first neural dataset contains one row per regular-season possession with
exactly one lineup segment. A possession with any substitution boundary is
excluded in full; the builder does not select a starting lineup, terminal
lineup, or fractional allocation.

For 2025-26 this retains **218,810 of 245,772 possessions** and excludes
**26,962**, or **10.970%**. See [Possessions](../data/possessions.md) for the
allocation audit.

Every row is oriented by possession:

- five offense player IDs;
- five defense player IDs;
- a home-offense sign equal to \(+1\) for home offense and \(-1\) for away
  offense;
- the target \(m_i\), offense points minus defense points.

The defense-points term preserves unusual possessions where the nominal
defense receives points.

## Additive boundary

The first PyTorch model has one scalar embedding \(e_p\) per player:

\[
\widehat{m_i}
= \mu
+ h s_i
+ \sum_{p \in O_i} e_p
- \sum_{p \in D_i} e_p
\]

Here \(\mu\) is average possession scoring margin, \(h\) is the centered
home-offense effect, and \(s_i\) is the home-offense sign. Player embeddings
start at zero. Embedding row zero is reserved for an unknown player.

Because adding the same constant to every player leaves a five-on-five
prediction unchanged, published embeddings are centered after training. The
reported one-number value is

\[
\text{Neural RAPM}_p = 200(e_p - \overline{e}).
\]

The factor 200 converts a player embedding to net-rating units over two
role-swapped possessions. It does not make the possession-level objective
identical to the stint-weighted ridge objective, so rankings and predictions
should be compared rather than requiring coefficient equality.

## Evaluation boundary

The model reuses game-date-safe expanding splits:

1. Every learning-rate and weight-decay candidate trains with early stopping
   on each expanding validation fold.
2. Validation-possession-weighted MSE across folds selects the candidate.
3. The winner's best epoch from the most recent fold sets the refit duration.
4. A fresh model trains on every final-train game for that duration.
5. The untouched final 15% of games produces mean and additive-neural test
   metrics.
6. Another fresh model trains on the full season for the public ranking and
   reusable checkpoint.

Reported test metrics are possession RMSE, MAE, skill relative to the training
mean, and game-margin RMSE after converting every prediction back to the home
perspective.

The regular holdout and playoffs are excluded from all hyperparameter and epoch
selection. Search trials and weighted candidate summaries are retained as
model artifacts.

### 2025-26 exemplar

The current Leaderboard exemplar is
`neural-2025-26-20260729T173539Z-51bc0264`. Its 20-candidate, three-fold search
selected:

| Setting | Value |
| --- | ---: |
| Learning rate | `0.0003` |
| AdamW weight decay | `0.001` |
| Refit epochs | 3 |
| Weighted validation MSE | 1.432444865 |

Learning rate was consequential, but the decay surface was nearly flat:
decays `0`, `0.001`, and `0.01` at learning rate `0.0003` differed by less
than \(10^{-7}\) weighted validation MSE. The exact selected decay should not
be interpreted as evidence that `0.001` is materially better than its
neighbors.

## Transformer token contract

The intended first attention sequence has 13 tokens:

| Token | Contents |
| --- | --- |
| `[STATE]` | Pre-possession game context |
| `[OFFENSE]` | Offense role marker |
| Five player tokens | Shared player embeddings plus offense-role embeddings |
| `[DEFENSE]` | Defense role marker |
| Five player tokens | Shared player embeddings plus defense-role embeddings |

There will be no position encoding within either five-player set. Shuffling
players within offense or defense must leave the prediction unchanged. Stable
player identity and player-season or bio features remain separate inputs so
the model can eventually represent new or low-exposure players.

Initial state features may include home role, score differential, period,
remaining time, and postseason status. Every state value must be available
before the possession begins.

## Interpretation

Attention weights will not be treated as player attribution. Planned
interpretation outputs include:

- role-swapped lineup predictions;
- one-player replacement counterfactuals;
- interaction residuals relative to the additive model;
- permutation or Shapley-style lineup contributions;
- uncertainty and stability across seeds and seasons.

## Correctness tests

`tests/test_neural_modeling.py` enforces the initial contract:

- multi-segment possessions are excluded rather than allocated;
- home and away possessions are oriented correctly;
- player index zero remains reserved;
- player order within a lineup cannot affect additive predictions;
- swapping offense and defense reverses the signed player contribution;
- the grid search evaluates every candidate on every requested fold and marks
  exactly one weighted-MSE winner;
- the miniature Lightning experiment emits selection, final-test, and
  all-season checkpoints plus search evidence, metrics, and rankings.

Run the focused checks with:

```bash
uv run pytest -q tests/test_neural_modeling.py
```

Training commands and artifact descriptions are in
[Train neural models](../guides/train-neural.md). The current regular-holdout
and playoff results are tracked on the shared
[Leaderboard](leaderboard.md) scoreboard.
