# Prompt: 将当前 UniSAR 项目原地改造成 RelSAR

你是 coding agent。这个 README 会直接放在原始 `UniSAR-main/` 项目根目录中。请把**当前项目**原地改造成 RelSAR：基于原始 UniSAR 的数据接口、训练流程和预测头，加入下面定义的轻量机制。目标不是从其他目录复制一个复杂版 RelSAR，而是在当前 UniSAR 代码上做最小必要改动，得到一个更简洁、逻辑清楚的 RelSAR 实现。

## Behavioral Guardrails

这些规则优先于速度。任务很小的时候可以用判断力，但默认要偏谨慎、偏简洁。

### 1. Think Before Coding

不要假设，不要隐藏困惑。实现前必须先做三件事：

- 明确说出你对当前项目结构和目标 RelSAR 方案的假设。
- 如果存在多个合理解释，先列出来，不要静默选择。
- 如果某个地方不清楚，停下来说明哪里不清楚，并提出具体问题。

本任务中特别要确认：

- 是否只改 `models/UniSAR.py` 和两个 yaml。
- 是否保留原始 `q_i_cl_loss` 与 `his_cl_loss`。
- 是否保持 counterfactual 只影响 attention bias，不影响 belief update。

### 2. Simplicity First

写能解决问题的最少代码，不做投机性功能。

- 不添加本文档没有要求的模块、loss、diagnostics 或配置。
- 不为了“灵活性”加入额外开关。
- 不为单次使用代码抽复杂抽象。
- 如果一段实现可以 50 行完成，不要写成 200 行。

本任务中的简洁边界：

- belief 是 Bayesian online update with drift，不做 temperature。
- counterfactual 是 source-removal posterior attribution，不重跑 prediction head。
- attention bias 是 posterior similarity + sigma confidence + source gate，不加额外 attention trick。

### 3. Surgical Changes

只改必须改的地方。每一行 diff 都应该能对应到本文档某个 Level。

- 不顺手重构无关代码。
- 不改 dataloader、sampler、Runner，除非实现无法运行且必须改。
- 不改原始 PLE prediction head 和 loss 接口。
- 不删除原本存在但与本任务无关的 dead code；可以在最终说明里指出。
- 如果你的改动造成了未使用 import、变量或函数，清理你自己造成的 orphan。

### 4. Goal-Driven Execution

实现前给出简短计划，每一步都要有验证方式：

```text
1. Add multi-intent discovery -> verify tensor shapes [B, L, K]
2. Add Bayesian belief trace with drift -> verify posterior/confidence shapes and no NaN
3. Add source-removal attribution -> verify KL effects and gate shape [B, L, 2]
4. Replace Transformer attention -> verify logits accept pair-wise bias and masks still work
5. Wire into forward/loss/config -> verify py_compile and a minimal training startup
```

不要只说“make it work”。成功标准以第 9 节验收标准为准。

```text
Multi-Intent Discovery
 -> Bayesian Sequential Intent Belief Update with Drift
 -> Source-Removal Counterfactual Attribution
 -> Confidence-Calibrated Intent/Source Matching Bias in Attention
 -> 原 UniSAR prediction heads
```

核心原则：

- 保留原始 UniSAR 的数据接口、loss 接口、PLE prediction head、query-item alignment、history contrastive loss。
- 默认保留 `models/UniSAR.py`、`class UniSAR` 和 `--model UniSAR` 训练入口，先把实现原地改成 RelSAR；除非后续明确要求全量重命名，否则不要为了项目命名破坏现有加载逻辑。
- 不做 belief-to-temperature，不引入 uncertainty attention temperature。
- 不做 heavy counterfactual forward，也就是不要为 full / wo_cross / wo_same 重复跑 prediction head。
- Counterfactual attribution 只能影响 attention bias，不能反过来决定 belief update。
- 改动尽可能集中在 `models/UniSAR.py` 和两个 yaml 配置；除非必要，不改 dataloader、sampler、Runner。

## 0. 先阅读并确认当前项目结构

必须先读这些文件：

- `models/UniSAR.py`
- `config/UniSAR_KuaiSAR.yaml`
- `config/UniSAR_Amazon.yaml`
- `main.py`
- `utils/Runner.py`

原始 `models/UniSAR.py` 的结构大致是：

- `UniSAR.parse_model_args`
  - 只有 `num_layers / num_heads`
  - `q_i_cl_*`
  - `his_cl_*`
  - `pred_hid_units`
- `UniSAR.__init__`
  - 三个 Transformer：
    - `rec_transformer`
    - `src_transformer`
    - `global_transformer`
  - 两个 cross decoder：
    - `rec_cross_fusion`
    - `src_cross_fusion`
  - 两个 target attention pooling：
    - `rec_his_attn_pooling`
    - `src_his_attn_pooling`
  - PLE + rec/src tower
- `forward`
  - 先构造 `all_his_emb`
  - `global_transformer` 用 `global_mask` 得到 cross-domain encoded states
  - `rec_transformer/src_transformer` 得到 same-domain states
  - cross decoder 融合
  - target attention pooling 得到 `rec_fusion/src_fusion`
- `rec_loss/src_loss/rec_predict/src_predict`
  - 接口不要破坏
- `Transformer`
  - 当前是 PyTorch `nn.TransformerEncoder`
  - 不能直接加 per-sample attention bias，因此需要替换成轻量自定义 attention

## 1. Level 1: Multi-Intent Discovery

### 目标

为用户历史行为学习多个 latent interests，并给每个行为 token 一个初始 intent 分布。

### 必改位置

文件：`models/UniSAR.py`

### 新增参数

在 `parse_model_args` 增加：

```python
parser.add_argument('--intent_num', type=int, default=8)
parser.add_argument('--intent_heads', type=int, default=2)
parser.add_argument('--intent_dropout', type=float, default=0.1)
parser.add_argument('--intent_temp', type=float, default=0.5)
parser.add_argument('--intent_var_min', type=float, default=1e-4)
parser.add_argument('--intent_diversity_weight', type=float, default=0.01)
parser.add_argument('--intent_diversity_margin', type=float, default=0.2)
```

只保留这些最小参数。不要加入 `intent_entropy_target`、`intent_confidence_target` 这类额外目标，除非后续实验明确需要。

### 新增模块

新增 `LatentIntentDiscovery(nn.Module)`：

- learnable `intent_slots: [K, D]`
- 用 `nn.MultiheadAttention` 让 slots attend 到 behavior sequence
- 输出 `intents: [B, K, D]`
- 处理全 mask 行，避免 attention NaN

### 新增函数

在 `UniSAR` 中新增：

```python
compute_intent_state(seq_emb, seq_mask)
```

逻辑：

```text
intents = latent_intent_discovery(seq_emb, seq_mask)
assign_logits = seq_emb @ intents.T / intent_temp
prior_assign = softmax(assign_logits)
masked positions set to 0
```

返回：

```text
intents: [B, K, D]
prior_assign: [B, L, K]
intent_reg: scalar
diagnostics: minimal dict
```

`intent_reg` 只做一个轻量 diversity regularization：

```text
normalize intents
off_diag_sim = cosine similarity between different intent slots
intent_reg = mean(relu(off_diag_sim - margin))
```

不要加入复杂 usage entropy / confidence target，避免主机制被 regularization 淹没。

## 2. Level 2: Bayesian Sequential Intent Belief Update with Drift

### 目标

belief 不做 temperature。belief 是一个贝叶斯在线更新模块：每来一个新行为 `x_t`，把它当成 evidence，和上一时刻所有 interest belief 分布比较，计算 posterior responsibility，然后更新每个 interest 的 Gaussian belief state。

同时加入非常轻的动态兴趣漂移：旧 evidence 通过 decay factor `rho` 衰减，新行为可以逐步推动 `mu/var/mass` 漂移。这个 drift 只属于 belief update，不属于 counterfactual gate。

### 核心定义

每个 interest 维护：

```text
mu_k    : 当前 interest Gaussian 均值
var_k   : 当前 interest Gaussian 方差
mass_k  : 当前 interest 已吸收的有效 evidence mass
sigma_k : sqrt(var_k)，用于 confidence calibration
```

把每个 intent 看成一个 latent hypothesis：

```text
H_k = 第 k 个 latent interest
B_{k,t-1} = N(mu_{k,t-1}, sigma^2_{k,t-1})
x_t = 当前行为 evidence
r_{t,k} = p(z_t = k | x_t, B_{t-1})
```

每个 time step `t` 必须按下面顺序实现。

Step 1: 判断 token 是否有效。

```text
valid_t = not seq_mask[:, t]
can_update_t = valid_t & update_mask[:, t]  # update_mask=None 时等于 valid_t
```

padding token：

- posterior 写 0；
- confidence 写 0；
- 不更新 `mu/var/mass`。

Step 2: 用上一时刻 belief 计算 likelihood。

```text
delta_k = x_t - mu_k
cost_k = mean(delta_k^2 / clamp(var_k, var_min))
log_likelihood_k = -0.5 * cost_k
```

不需要完整 Gaussian 常数项；为了简洁，保留 Mahalanobis-style distance 即可。

Step 3: 结合 multi-intent prior 得到 posterior responsibility。

```text
log_prior_k = log(clamp(prior_assign_tk, 1e-8))
score_k = log_likelihood_k + belief_prior_weight * log_prior_k
r_t = softmax(score)
```

这里的 `r_t` 就是当前行为属于所有 interest 的 posterior distribution，也写入 `posterior_trace[:, t, :]`。

Step 4: 用当前 belief 的 sigma 计算 confidence。

```text
expected_sigma_t = sum_k r_tk * mean(sqrt(var_k))
belief_confidence_t = 1 / (1 + expected_sigma_t)
```

写入 `confidence_trace[:, t]`。这个 confidence 后面用于 pair-wise attention bias，不用于改变 update 规则。

Step 5: 对允许更新的 token 做 drift-aware Bayesian update。

设：

```text
rho = belief_drift_decay
old_mass = mass
old_mu = mu
old_var = var
effective_old_mass = rho * old_mass
update_weight = r_t if can_update_t else 0
new_mass = effective_old_mass + update_weight
```

均值更新：

```text
new_mu =
    (effective_old_mass * old_mu + update_weight * x_t)
    / clamp(new_mass, 1e-8)
```

二阶矩更新：

```text
old_second = old_var + old_mu^2
new_second =
    (effective_old_mass * old_second + update_weight * x_t^2)
    / clamp(new_mass, 1e-8)
new_var = clamp(new_second - new_mu^2, min=var_min)
```

只对 `can_update_t=True` 的样本行应用 `new_mass/new_mu/new_var`。对 `can_update_t=False` 的样本行，保持上一时刻 `mass/mu/var` 不变；不要让 source-removal counterfactual 的 removed source 更新 belief。

Step 6: 进入下一步 `t+1`。

```text
mu, var, mass = updated belief state
```

这个循环给模型明确的“收敛/漂移”感觉：

- 同一类 evidence 持续出现时，`mu_k` 逐步稳定，`var_k` 下降或维持稳定；
- 新 evidence 长期偏离时，`rho < 1` 让旧 mass 衰减，interest 可以漂移；
- `var_k/sigma_k` 越大，后续 attention bias 的 confidence 越低。

### 新增参数

```python
parser.add_argument('--belief_init_var', type=float, default=1.0)
parser.add_argument('--belief_init_mass', type=float, default=1.0)
parser.add_argument('--belief_prior_weight', type=float, default=1.0)
parser.add_argument('--belief_drift_decay', type=float, default=0.98)
```

`belief_drift_decay` 建议范围：

```text
1.00 : 累积式 Bayesian update，没有动态漂移
0.98 : 轻微漂移，推荐默认
0.95 : 更强调近期行为，但可能不稳定
```

不要加入：

- `attention_base_temp`
- `uncertainty_temp_scale`
- `transformer_temp_min`
- `transformer_temp_max`
- `use_uncertainty_attention`
- `uncertainty_fusion`

这些都属于旧的 belief-to-temperature 方案，本方案不要。

### 新增函数

```python
compute_belief_trace(seq_emb, intents, prior_assign, seq_mask, update_mask=None)
```

参数：

- `seq_emb: [B, L, D]`
- `intents: [B, K, D]`
- `prior_assign: [B, L, K]`
- `seq_mask: [B, L]`，True 表示 padding
- `update_mask: [B, L] | None`，True 表示这个 token 允许更新当前 belief state

返回：

```text
posterior_trace: [B, L, K]
confidence_trace: [B, L]
final_mu, final_var, final_mass
minimal diagnostics
```

注意：

- 即使 `update_mask[:, t]` 为 False，也可以计算 `posterior_t`，因为 source-removal counterfactual 需要“在去掉某来源时，当前 token 会如何匹配剩余 belief”。
- 但只有 `update_mask[:, t]` 为 True 的 token 才能更新 `mu/var/mass`。
- `confidence_trace` 来自 `var/sigma`，只用于 attention bias 的可靠性校准，不用于决定是否 update。
- padding token 既不计算有效 posterior，也不更新。

## 3. Level 3: Lightweight Source-Removal Counterfactual Attribution

### 目标

用非常轻的 counterfactual 解释当前行为或当前 intent 判断更依赖之前的 search 历史还是 recommend 历史。

这不是 prediction-level heavy counterfactual，不重跑 prediction head。只在 belief/posterior 层做 source-removal ablation。

### 关键原则

必须保持这个方向：

```text
belief update 决定 counterfactual attribution
counterfactual attribution 再影响 attention bias
```

不要写成：

```text
counterfactual attribution 决定 belief update
```

否则会变成循环自证。

### 维护三套 belief

对 `all_his_emb` 维护：

```text
full belief   : rec + search 都更新
no_rec belief : 只用 search 更新，相当于移除 recommend history
no_src belief : 只用 recommend 更新，相当于移除 search history
```

其中 `all_his_type` 约定：

```text
1 = recommend behavior
2 = search behavior
```

构造：

```python
valid = ~all_his_mask
rec_update = valid & (all_his_type == 1)
src_update = valid & (all_his_type == 2)
full_update = valid
no_rec_update = src_update
no_src_update = rec_update
```

调用：

```python
full_belief = compute_belief_trace(..., update_mask=full_update)
no_rec_belief = compute_belief_trace(..., update_mask=no_rec_update)
no_src_belief = compute_belief_trace(..., update_mask=no_src_update)

p_full = full_belief.posterior_trace
p_no_rec = no_rec_belief.posterior_trace
p_no_src = no_src_belief.posterior_trace
full_confidence = full_belief.confidence_trace
```

### 计算 source-removal effect

使用分布差异：

```text
rec_effect_t = KL(p_full_t || p_no_rec_t)
src_effect_t = KL(p_full_t || p_no_src_t)
```

解释：

- `rec_effect_t` 大：去掉 recommend history 后 full posterior 变化大，说明 recommend history 对当前 intent 判断重要。
- `src_effect_t` 大：去掉 search history 后变化大，说明 search history 重要。

实现细节：

```python
def symmetric_or_forward_kl(p, q):
    p = p.clamp_min(1e-8)
    q = q.clamp_min(1e-8)
    return (p * (p.log() - q.log())).sum(dim=-1)
```

最小实现用 forward KL 即可，不需要 symmetric KL。

然后：

```python
cf_logits = torch.stack([rec_effect, src_effect], dim=-1)
cf_source_gate = torch.softmax(cf_logits / cf_temp, dim=-1)
```

新增参数：

```python
parser.add_argument('--cf_temp', type=float, default=1.0)
parser.add_argument('--cf_bias_scale', type=float, default=1.0)
```

`cf_bias_scale` 只定义一次；这里列出是因为它由 counterfactual attribution 产生，并在 attention bias 中使用。

不要添加：

- `cf_gate_scale`
- `cf_consistency_weight`
- `rec_src_mix`
- `src_cross_mix`
- full / wo_cross / wo_same prediction branches

这些属于 heavy counterfactual，不符合当前轻量方案。

### 输出

`compute_source_counterfactual` 返回：

```text
cf_source_gate: [B, L, 2]  # index 0 = rec support, index 1 = search support
cf_rec_effect: [B, L]
cf_src_effect: [B, L]
diagnostics minimal
```

padding 位置置 0。

## 4. Level 4: Confidence-Calibrated Intent/Source Matching Bias in Attention

### 目标

让 belief posterior、belief sigma confidence、source-removal counterfactual attribution 只影响 attention bias。

这一步是 pair-wise attention bias：每个 query token `i` 和 key token `j` 都会得到一个额外 logit bias。

### 为什么要替换 Transformer

原始 `Transformer` 用 `nn.TransformerEncoder`，不方便加入 batch-specific attention bias。需要替换成轻量自定义实现：

- `IntentSourceSelfAttention`
- `IntentSourceTransformerLayer`
- `Transformer`

不要引入 temperature，不要引入 attention probability power。

### Attention logits

基础：

```text
attn_logits = QK^T / sqrt(d)
```

Intent matching bias：

```text
intent_sim_ij = p_i @ p_j
intent_center = 1 / intent_num
raw_intent_bias_ij = intent_sim_ij - intent_center
```

Sigma confidence calibration：

```text
confidence_i = belief_confidence_trace[:, i]
confidence_j = belief_confidence_trace[:, j]
pair_confidence_ij = sqrt(confidence_i * confidence_j)
intent_bias_ij = raw_intent_bias_ij * pair_confidence_ij
```

解释：

- posterior 相似但 sigma 大，说明 belief 不确定，bias 应该变弱；
- posterior 相似且 sigma 小，说明两个 token 的 interest 判断可靠，bias 才强；
- confidence 不改变 posterior，也不改变 belief update，只校准 attention bias。

Source counterfactual bias：

对 query token `i`，它有：

```text
cf_source_gate_i = [rec_support_i, src_support_i]
```

对 key token `j`，根据 `all_his_type_j`：

```text
source_bias_ij = rec_support_i if key j is recommend
source_bias_ij = src_support_i if key j is search
source_bias_ij = 0 for padding
```

中心化：

```text
raw_source_bias_ij = source_bias_ij - 0.5
source_bias_ij = raw_source_bias_ij * confidence_i
```

这里用 query token `i` 的 confidence 即可，因为 source attribution 是 query-conditioned：它回答的是“当前 token `i` 的 intent 判断更该依赖哪类历史来源”。

最终：

```text
attn_logits += intent_bias_scale * intent_bias
attn_logits += cf_bias_scale * source_bias
```

新增参数：

```python
parser.add_argument('--intent_bias_scale', type=float, default=1.0)
```

### Transformer forward 签名

把原始：

```python
forward(his_emb, src_key_padding_mask, src_mask=None)
```

改成：

```python
forward(
    his_emb,
    src_key_padding_mask,
    src_mask=None,
    intent_assign=None,
    belief_confidence=None,
    source_gate=None,
    token_type=None,
)
```

其中：

- `intent_assign: [B, L, K]`
- `belief_confidence: [B, L]`
- `source_gate: [B, L, 2]`
- `token_type: [B, L]`

局部 `rec_transformer/src_transformer` 可以传 `intent_assign/belief_confidence`，不传 `source_gate`。全局 `global_transformer` 同时传 `intent_assign/belief_confidence/source_gate/token_type`。

### Mask 逻辑

保留原始 mask 语义：

- `src_key_padding_mask`: True 表示 padding
- `src_mask`: True 表示不能 attend

注意原始 `global_mask = all_his_type[:, :, None] == all_his_type[:, None, :]`，True 表示 mask same-domain，只保留 cross-domain attention。不要改变这个语义，除非你明确想改模型结构。

## 5. Forward 中的最小接入方式

在 `UniSAR.forward` 中，推荐按下面顺序改：

```text
1. user_emb = get_user_emb(user)
2. all_his_emb, all_his_mask, q_i_align_used = get_all_his_emb(...)
3. rec_his_mask / src_his_mask
4. all_intents, all_prior_assign, intent_reg = compute_intent_state(all_his_emb, all_his_mask)
5. full_belief = compute_belief_trace(all_his_emb, all_intents, all_prior_assign, all_his_mask, full_update)
6. no_rec_belief = compute_belief_trace(..., no_rec_update)
7. no_src_belief = compute_belief_trace(..., no_src_update)
8. full_posterior = full_belief.posterior_trace
9. full_confidence = full_belief.confidence_trace
10. cf_source_gate = KL-based source-removal attribution
11. global_transformer(..., intent_assign=full_posterior, belief_confidence=full_confidence, source_gate=cf_source_gate, token_type=all_his_type)
12. split global encoded into src2rec / rec2src
13. split full_posterior into rec_posterior / src_posterior
14. split full_confidence into rec_confidence / src_confidence
15. rec_transformer(..., intent_assign=rec_posterior, belief_confidence=rec_confidence)
16. src_transformer(..., intent_assign=src_posterior, belief_confidence=src_confidence)
17. keep original cross decoder fusion and target attention pooling
18. return user_feats, q_i_align_used, his_cl_used, regularization
```

为了最小改动，可以继续保留原始 cross decoder fusion：

```python
rec_fusion_decoded = self.rec_cross_fusion(...)
src_fusion_decoded = self.src_cross_fusion(...)
```

本方案的 counterfactual 不替换 fusion；它只调节 attention bias，特别是 global cross-domain attention。

## 6. Loss 和 diagnostics

### 保留

原始：

- `click_loss`
- `q_i_cl_loss`
- `his_cl_loss`
- `total_loss`

### 新增最小 regularization

只新增：

```text
intent_reg
```

加入 total loss：

```python
total_loss += intent_diversity_weight * intent_reg
```

### 新增最小 diagnostics

可以加到 `loss_dict`：

```text
intent_reg
belief_entropy_mean
belief_sigma_mean
belief_confidence_mean
cf_rec_effect_mean
cf_src_effect_mean
cf_rec_gate_mean
cf_src_gate_mean
```

不要复制当前复杂版本里的大量 diagnostics，比如：

- early/mid/late uncertainty
- attention temperature min/max
- rec/src mix
- full/wo_cross/wo_same delta
- cf consistency

这些不属于当前简洁方案。

建议把重复的 auxiliary loss 逻辑抽成：

```python
add_auxiliary_losses(...)
finalize_loss_dict(...)
```

但这属于代码清理，不是核心机制；如果担心改动范围，可以暂时不抽。

## 7. 配置文件改动

修改：

- `config/UniSAR_KuaiSAR.yaml`
- `config/UniSAR_Amazon.yaml`

新增最小配置：

```yaml
intent_num: 8              # Amazon 可用 16，但先统一 8 更稳
intent_heads: 2
intent_dropout: 0.1
intent_temp: 0.5
intent_var_min: 0.0001
intent_diversity_weight: 0.01
intent_diversity_margin: 0.2

belief_init_var: 1.0
belief_init_mass: 1.0
belief_prior_weight: 1.0
belief_drift_decay: 0.98

intent_bias_scale: 1.0
cf_temp: 1.0
cf_bias_scale: 1.0
```

不要加入 temperature/uncertainty 配置。

## 8. 明确不要做的事

不要做这些：

1. 不要把 belief entropy 转成 attention temperature。
2. 不要加 `uncertainty_fusion`。
3. 不要做 prediction-level counterfactual：
   - `full_pred`
   - `wo_cross_pred`
   - `wo_same_pred`
4. 不要加 learnable `rec_src_mix/src_cross_mix`。
5. 不要让 source counterfactual gate 决定 belief 是否 update。
6. 不要让 sigma/confidence 决定 belief 是否 update；它只用于 attention bias calibration。
7. 不要大改 Runner、dataset、sampler。
8. 不要把其他目录里的复杂 RelSAR 版本直接复制过来；它可能包含当前方案不需要的额外机制。

## 9. 验收标准

实现完成后至少检查：

```bash
python -m py_compile models/UniSAR.py
```

如果环境允许，跑一个最小训练启动命令：

```bash
python3 main.py --model UniSAR --data KuaiSAR
```

看日志里至少包含：

- `click_loss`
- 如果开启原辅助 loss：`q_i_cl_loss`、`his_cl_loss`
- `intent_reg`
- `belief_entropy_mean`
- `belief_sigma_mean`
- `belief_confidence_mean`
- `cf_rec_effect_mean`
- `cf_src_effect_mean`
- `cf_rec_gate_mean`
- `cf_src_gate_mean`
- `total_loss`

形状检查：

```text
all_prior_assign: [B, L_all, K]
full_posterior: [B, L_all, K]
full_confidence: [B, L_all]
cf_source_gate: [B, L_all, 2]
rec_posterior: [B, L_rec, K]
rec_confidence: [B, L_rec]
src_posterior: [B, L_src, K]
src_confidence: [B, L_src]
global transformer output: [B, L_all, D]
rec/src transformer output: [B, L_rec/src, D]
```

语义检查：

- `full belief` 用全部非 padding 历史更新。
- `no_rec belief` 只用 search 历史更新。
- `no_src belief` 只用 recommend 历史更新。
- `cf_rec_effect = KL(full || no_rec)`。
- `cf_src_effect = KL(full || no_src)`。
- belief update 使用 `rho = belief_drift_decay` 衰减旧 mass，实现动态兴趣漂移。
- `belief_confidence = 1 / (1 + expected_sigma)`，只校准 attention bias。
- intent bias 是 pair-wise posterior similarity，并乘以 `sqrt(conf_i * conf_j)`。
- source bias 是 query-conditioned source support，并乘以 `conf_i`。
- counterfactual attribution 只进入 attention bias。
- belief update 不依赖 counterfactual gate。

## 10. 推荐最终叙事

代码实现应支持下面这条清晰叙事：

```text
Multi-intent discovery learns latent user interests from mixed search/recommend histories.
Bayesian sequential belief update treats each new behavior as evidence, computes posterior responsibility under Gaussian intent beliefs, and updates interest states online with a light drift factor.
Source-removal counterfactual attribution estimates whether each intent transition is mainly supported by previous search or recommendation evidence.
Confidence-calibrated intent/source matching bias injects posterior similarity, sigma-based reliability, and source-removal attribution into pair-wise self-attention.
```

中文表述：

```text
多兴趣模块给出初始兴趣空间；
贝叶斯顺序 belief update 把每个新行为视为 evidence，根据 Gaussian interest belief 计算 posterior responsibility，并通过 drift factor 允许用户兴趣动态漂移；
轻量 source-removal counterfactual 比较 full belief 与去掉 search/recommend 后的 posterior 差异，估计当前行为更由哪类历史支持；
最后把 posterior 相似度、sigma confidence 和归因结果一起用于 pair-wise attention bias，不反向决定 belief update。
```
