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
| Deep Sets | Nonlinear pooled sets | Nonlinear lineup strength without pairwise attention | Implemented |
| RAPM + Transformer | Self-attention residual | Player-player and role-dependent interactions around frozen RAPM | Implemented |

Deep Sets is an important ablation. If it matches RAPM + Transformer, the
benefit comes from nonlinear aggregation rather than attention specifically. This
follows the permutation-invariant construction in
[Deep Sets](https://arxiv.org/abs/1703.06114). The attention model follows
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

## Deep Sets architecture

Deep Sets preserves order invariance while adding a nonlinear lineup
representation. Player identity and role are concatenated, not added:

\[
t_{p,r} = [E_p;R_r],
\]

where \(E_p \in \mathbb{R}^{32}\) is a shared player embedding and
\(R_r \in \mathbb{R}^{8}\) is an offense or defense role embedding. A shared
player MLP transforms every token independently:

\[
\phi:\mathbb{R}^{40}\rightarrow\mathbb{R}^{64}\rightarrow\mathbb{R}^{64}.
\]

The five transformed tokens are summed separately by role:

\[
z_O=\sum_{p\in O_i}\phi(t_{p,O}),
\qquad
z_D=\sum_{p\in D_i}\phi(t_{p,D}).
\]

Each pooled vector has 64 elements. Concatenating offense, defense, and the
home-offense sign produces the 129-element lineup input:

\[
u_i=[z_O;z_D;s_i]\in\mathbb{R}^{64+64+1}.
\]

The lineup MLP is

\[
\rho:\mathbb{R}^{129}\rightarrow\mathbb{R}^{128}
\rightarrow\mathbb{R}^{64}\rightarrow\mathbb{R}.
\]

The final prediction nests the additive model:

\[
\widehat{m_i}
=
\mu + hs_i
+ \sum_{p\in O_i}a_p-\sum_{p\in D_i}a_p
+ \rho(u_i).
\]

The first four terms are the additive skip path. The final term is the
nonlinear lineup residual. The residual output layer starts at zero, so the
larger architecture begins at the additive-compatible boundary. All
parameters then train jointly.

This decomposition is operational, not identified. The Deep Sets branch can
also represent additive player effects, so its residual cannot be interpreted
as a uniquely estimated interaction effect or a causal quantity. It is useful
for inspecting how this fitted network allocated signal between its two paths.

### Tensor flow

For batch size \(B\):

| Stage | Shape |
| --- | --- |
| Offense and defense player IDs | \(B\times5\) each |
| Player embeddings | \(B\times5\times32\) |
| Concatenated player-role tokens | \(B\times5\times40\) |
| Player MLP outputs | \(B\times5\times64\) |
| Separate sum pools | \(B\times64\) each |
| Lineup MLP input | \(B\times129\) |
| Possession prediction | \(B\) |

Because only sums reach the lineup MLP, shuffling players within offense or
defense cannot change the result. Offense and defense remain distinguishable
through both the role embedding and their separate positions in \(u_i\).

### Training protocol

The 2025-26 run used a CPU-specific 12-candidate grid:

- learning rates `0.0003`, `0.001`, and `0.003`;
- AdamW weight decays `0`, `0.001`, `0.01`, and `0.1`;
- batch size 8,192;
- at most 15 epochs with patience five;
- the same three expanding validation folds as every other model.

After hyperparameter selection, seeds 17, 18, and 19 were refit independently
on the final-training and full-season samples. Seed 17 was designated as the
Leaderboard seed before holdout evaluation; the better holdout seed was not
substituted after results were observed.

### 2025-26 result

Run `deep-sets-2025-26-20260729T215128Z-dc12dd11` contains 51,002 trainable
parameters. Validation selected learning rate `0.001`, zero weight decay, and
one refit epoch.

| Seed | Possession RMSE | Game-margin RMSE |
| ---: | ---: | ---: |
| **17 (Leaderboard)** | 1.199759 | 15.1073 |
| 18 | 1.199676 | 15.0278 |
| 19 | 1.199548 | 14.8214 |

The canonical nonlinear residual had standard deviation only **0.000127**
points per possession. It was effectively a constant adjustment, so this run
did not learn meaningful lineup interactions before validation stopped
improving. This does not imply that lineup interactions are absent; it says
this specification and training protocol found no out-of-sample benefit from
using its nonlinear capacity.

Against additive neural, the paired game-cluster bootstrap found:

| Cohort | Metric | Deep Sets - additive | 95% interval |
| --- | --- | ---: | ---: |
| Regular holdout | Possession RMSE | +0.000306 | [+0.000170, +0.000452] |
| Regular holdout | Game-margin RMSE | +0.389153 | [+0.173284, +0.637469] |
| Playoffs | Possession RMSE | +0.000203 | [+0.000009, +0.000388] |
| Playoffs | Game-margin RMSE | -0.006494 | [-0.317953, +0.326210] |

The first Deep Sets specification therefore does **not** improve on the
additive model. The playoff game-margin difference is indistinguishable from
zero. This is still a useful architectural ablation: nonlinear pooled sets did
not earn predictive support under the current one-season, player-ID-only
protocol.

## RAPM + Transformer architecture

The first attention model treats canonical one-year ridge RAPM as a frozen
base learner and asks the Transformer to learn only an offense-oriented
possession residual:

\[
\widehat y_i
=
\widehat y_i^{RAPM}
+
f_\theta(O_i,D_i,s_i).
\]

The base prediction comes from the stint-weighted ridge model, converted to
the possession target as described in
[RAPM Base Predictions](../data/rapm-base-predictions.md). The Transformer
cannot change RAPM's coefficients or intercept. Its final linear layer is
initialized to zero, so before optimization
\(f_\theta(\cdot)=0\) and the combined prediction equals RAPM exactly.

### Token sequence

Every possession is encoded as 13 width-32 tokens:

| Position | Token contents |
| ---: | --- |
| 0 | Learned `[STATE]` token plus projected home-offense sign |
| 1 | Learned `[OFFENSE]` marker |
| 2-6 | Shared player embedding plus offense-role embedding |
| 7 | Learned `[DEFENSE]` marker |
| 8-12 | Shared player embedding plus defense-role embedding |

Player and role embeddings are **added**, so every token remains
32-dimensional. The same player table is used on both sides of the ball;
offense and defense role embeddings tell the encoder which role that player
currently occupies.

No positional encoding is added. A Transformer encoder without positional
information is permutation equivariant, and the model reads only the
`[STATE]` output. Consequently, shuffling the five offense players or five
defense players leaves the prediction unchanged. The two groups remain
distinguishable through their role embeddings and role-marker tokens.

#### Literal width-two example

For illustration, reduce the learned width from 32 to two. Let the role
embeddings be

\[
R_O=[1,0], \qquad R_D=[0,1].
\]

Suppose A-E are on offense and F-J are on defense, with toy player embeddings

```text
A=[0.1,0.1]  B=[0.2,0.1]  C=[0.3,0.1]  D=[0.4,0.1]  E=[0.5,0.1]
F=[0.1,0.2]  G=[0.2,0.2]  H=[0.3,0.2]  I=[0.4,0.2]  J=[0.5,0.2]
```

Let the learned special tokens be

```text
STATE=[0.1,0.2]  OFFENSE=[0.8,0.0]  DEFENSE=[0.0,0.8]
```

and let the state projection of a \(+1\) home-offense sign be
\([0.3,-0.1]\). The exact Transformer input is then:

| Row | Identity | Calculation | Token |
| ---: | --- | --- | --- |
| 0 | `[STATE]` | `[0.1,0.2] + [0.3,-0.1]` | `[0.4,0.1]` |
| 1 | `[OFFENSE]` | learned marker | `[0.8,0.0]` |
| 2 | A offense | `[0.1,0.1] + [1,0]` | `[1.1,0.1]` |
| 3 | B offense | `[0.2,0.1] + [1,0]` | `[1.2,0.1]` |
| 4 | C offense | `[0.3,0.1] + [1,0]` | `[1.3,0.1]` |
| 5 | D offense | `[0.4,0.1] + [1,0]` | `[1.4,0.1]` |
| 6 | E offense | `[0.5,0.1] + [1,0]` | `[1.5,0.1]` |
| 7 | `[DEFENSE]` | learned marker | `[0.0,0.8]` |
| 8 | F defense | `[0.1,0.2] + [0,1]` | `[0.1,1.2]` |
| 9 | G defense | `[0.2,0.2] + [0,1]` | `[0.2,1.2]` |
| 10 | H defense | `[0.3,0.2] + [0,1]` | `[0.3,1.2]` |
| 11 | I defense | `[0.4,0.2] + [0,1]` | `[0.4,1.2]` |
| 12 | J defense | `[0.5,0.2] + [0,1]` | `[0.5,1.2]` |

The role vectors are learned rather than fixed to these illustrative values.
They act as token-type labels: the same player moved from offense to defense
changes by \(R_D-R_O\). The frozen RAPM prediction remains outside this
matrix on the additive skip path.

The first specification uses:

| Component | Configuration |
| --- | --- |
| Player and role width | 32 |
| Attention heads | 4 |
| Encoder layers | 2 |
| Feedforward width | 128 |
| Activation | GELU |
| Dropout | `0.1` |
| Readout | LayerNorm, `32 -> 32 -> 1` residual MLP |

Only the home-offense sign enters the state token in this slice. Score
differential, period, remaining time, playoff status, player bios, and
player-season features are intentionally deferred. Every future state feature
must be known before the possession starts.

### Leakage-safe RAPM base

Each model-selection stage receives its own RAPM state:

1. `cv_0`, `cv_1`, and `cv_2` fit RAPM on that fold's training games.
2. Validation predictions use only the corresponding earlier training games.
3. `final` fits RAPM on 1,044 games and predicts the untouched 186-game
   regular-season holdout.
4. `all_season` fits RAPM on all 1,230 regular-season games for frozen playoff
   inference.

Training rows use residuals from the RAPM fit on the same training window.
Validation, regular-holdout, and playoff base predictions are out of sample.
The validation and test games never contribute to their own RAPM base state.

### Optimization protocol

The initial CPU search is deliberately bounded:

- learning rates `0.0003` and `0.001`;
- AdamW weight decays `0` and `0.01`;
- batch size 8,192;
- at most 10 epochs with patience three;
- validation-possession-weighted MSE across the same three expanding folds
  used by the other models.

After selection, seeds 17, 18, and 19 are refit independently on both the
final-training and all-season samples. Seed 17 is the predetermined
Leaderboard model; holdout results are not used to choose among seeds.

Detailed commands, tensor settings, and artifacts are in
[Train RAPM + Transformer](../guides/train-transformer.md).

### 2025-26 result

Run `rapm-transformer-2025-26-20260729T233233Z-e316a73e` contains 45,409
trainable parameters. The four-candidate search selected learning rate
`0.0003`, weight decay `0.01`, and one refit epoch. Zero and `0.01` weight
decay differed by only \(4.0\times10^{-8}\) weighted validation MSE at the
selected learning rate, so the precise decay winner is not substantive.

The untouched regular-season holdout produced:

| Model or seed | Possession RMSE | Game-margin RMSE | Residual mean | Residual SD |
| --- | ---: | ---: | ---: | ---: |
| Frozen ridge RAPM | 1.199460 | 14.7107 | - | - |
| **17 (Leaderboard)** | 1.199526 | 14.7182 | -0.002883 | 0.001312 |
| 18 | 1.199451 | 14.7183 | 0.000495 | 0.000676 |
| 19 | 1.199391 | 14.6934 | 0.002149 | 0.001639 |

Seed 17 was fixed before holdout evaluation. Seed 19 is retained as stability
evidence and is not substituted after observing its better result.

For the canonical seed, paired game-cluster bootstrap differences relative to
frozen ridge RAPM were:

| Cohort | Metric | Transformer - RAPM | 95% interval |
| --- | --- | ---: | ---: |
| Regular holdout | Possession RMSE | +0.000065 | [+0.000031, +0.000099] |
| Regular holdout | Game-margin RMSE | +0.007408 | [-0.002541, +0.016340] |
| Playoffs | Possession RMSE | +0.000169 | [+0.000051, +0.000285] |
| Playoffs | Game-margin RMSE | -0.009226 | [-0.042714, +0.026947] |

The model is worse than RAPM at possession level in both cohorts. Its
game-margin differences are too small to distinguish from zero. The canonical
holdout residual is also mostly a small global downward adjustment rather
than strong lineup-specific variation. This first specification therefore
does not provide predictive evidence for useful attention interactions.

### Architectural status

The 13-token run remains a reproducible ablation, but its special tokens are
not the cleanest test of attention:

- role embeddings are necessary because they label every player as offense or
  defense;
- `[OFFENSE]` and `[DEFENSE]` contain no observed information and act only as
  optional learned processing slots;
- `[STATE]` carries the home-offense sign and serves as a learned pooling
  query, but neither function requires a state token.

The next recommended ablation removes all three special tokens. It applies
self-attention to ten role-tagged player tokens, mean-pools the five encoded
offense and five encoded defense players separately, and predicts the
residual from

\[
u_i=[z_O;z_D;s_i].
\]

This makes the comparison with Deep Sets precise: Deep Sets transforms each
player independently before role-wise pooling, while the Transformer permits
player-player interaction before the same pooling boundary.

## Interpretation

Attention weights are not treated as player attribution. The first run stores
the frozen RAPM prediction, Transformer residual, combined prediction, and
observed-lineup residual summaries separately. Future interpretation outputs
include:

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
- concatenated role inputs produce 40-element Deep Sets player tokens;
- shuffling either Deep Sets lineup leaves predictions unchanged;
- stored Deep Sets predictions decompose into additive and nonlinear terms;
- swapping offense and defense reverses the signed player contribution;
- the grid search evaluates every candidate on every requested fold and marks
  exactly one weighted-MSE winner;
- the miniature Deep Sets experiment emits three predetermined final-test and
  all-season seed checkpoints plus search, stability, and interaction evidence.

`tests/test_transformer_modeling.py` additionally verifies:

- changing validation outcomes cannot change their fold-specific RAPM base
  predictions;
- every validation and test base prediction is marked out of sample;
- the zero-initialized Transformer exactly reproduces RAPM;
- shuffling players within either lineup leaves predictions unchanged;
- stored combined predictions exactly equal RAPM plus Transformer residual;
- the miniature experiment emits all three predetermined holdout and
  all-season checkpoints;
- unseen players use the reserved embedding row and are counted during
  inference.

Run the focused checks with:

```bash
uv run pytest -q \
  tests/test_neural_modeling.py \
  tests/test_transformer_modeling.py
```

Training commands and artifact descriptions are in
[Train neural models](../guides/train-neural.md) and
[Train RAPM + Transformer](../guides/train-transformer.md). The current
regular-holdout and playoff results are tracked on the shared
[Leaderboard](leaderboard.md) scoreboard.
