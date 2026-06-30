# 可观测性（LangSmith）

本项目的 RAG 链路通过 LangSmith 做全链路 tracing、turn 级 metadata、在线 Eval 与用户结构化反馈闭环。

## 启用方式

| 变量 | 说明 |
|------|------|
| `APP_ENV` | `dev` / `prod` / `test`；默认 project 为 `rag-{APP_ENV}` |
| `LANGSMITH_API_KEY` | **有 key 即默认开启 tracing** |
| `LANGSMITH_TRACING_ENABLED` | 可选；设为 `false` 可强制关闭 |
| `LANGSMITH_PROJECT` | 可选；覆盖默认 `rag-dev` / `rag-prod` |
| `LANGSMITH_ENDPOINT` | 自托管 LangSmith 端点（可选） |

本地开发示例：

```env
APP_ENV=dev
LANGSMITH_API_KEY=lsv2_pt_xxx
```

生产 Docker 示例：

```env
APP_ENV=prod
LANGSMITH_API_KEY=lsv2_pt_xxx
# 自动写入 project rag-prod
```

## Turn 级 Metadata 契约

写入 **root run** 的 `extra`（RunnableConfig `metadata` 为初始子集，stream 结束后 patch 完整 turn 数据）。

### 身份与 ACL

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_id` | string | JWT 用户 ID |
| `session_id` | string | LangGraph `thread_id` |
| `is_admin` | bool | 是否管理员（影响 ES ACL） |
| `env` | string | `APP_ENV` |
| `document_ids_in_session` | string[] | **本 turn** 检索命中的 document_id（去重） |

### 检索

| 字段 | 类型 | 说明 |
|------|------|------|
| `retrieval_plan.action` | `skip` \| `retrieve` | 检索路由决策 |
| `retrieval_plan.strategy` | `none` \| `multi_query` \| `hyde` \| `decompose` | 变换策略 |
| `retrieval_plan.reason` | string | 路由理由（中文） |
| `retrieval_plan.standalone_query` | string | 可选，改写后的检索 query |
| `chunk_refs` | object[] | `{document_id, chunk_index, ref_id, filename}`，最多 20 条 |
| `rerank_scores` | object[] | `{ref_id, score}` |

### Grounding

| 字段 | 类型 | 说明 |
|------|------|------|
| `grounding_status` | `supported` \| `partial` \| `not_supported` \| `skipped` | 答案支撑校验结果 |
| `supported_ratio` | float | 0–1 |

### Tags（LangSmith 过滤）

- `rag`, `rag_chat` / `rag_resume`
- `env:dev` / `env:prod`
- `admin` / `user`

## 用户反馈（结构化）

流式回答完成后，assistant 消息展示 👍 / 👎。仅当次 SSE 带有 `run_id` 时可反馈（history reload 暂无 `run_id`）。

```http
POST /api/chat/feedback
Authorization: Bearer <token>
Content-Type: application/json
```

**👍 有帮助**

```json
{
  "run_id": "<uuid>",
  "trace_id": "<uuid>",
  "kind": "thumbs_up",
  "session_id": "<session>"
}
```

LangSmith：`user_thumbs_up`，score = 1.0

**👎 不准确**（`reason` 必填）

```json
{
  "run_id": "<uuid>",
  "trace_id": "<uuid>",
  "kind": "thumbs_down",
  "reason": "hallucination",
  "comment": "可选补充说明",
  "session_id": "<session>"
}
```

| `reason` | 含义 |
|----------|------|
| `retrieval_wrong` | 检索不准/漏检 |
| `hallucination` | 编造/无依据 |
| `tool_error` | 工具调用问题 |
| `too_slow` | 响应太慢 |
| `other` | 其他 |

LangSmith：`user_thumbs_down`，score = 0.0，`extra.reason` 为上述 enum。

必须传 `trace_id`（可与 `run_id` 相同）以走后台批量上传。

---

## Phase A — A1：Alerts 操作清单

**目标**：`rag-prod` 异常自动通知 Slack / PagerDuty。

### 步骤 1：获取 project UUID

1. 打开 [LangSmith](https://smith.langchain.com) → 选择 `rag-prod`
2. **Settings** → 复制 Project / Session ID（UUID）
3. 写入运维环境（不进 git）：

```bash
export LANGSMITH_API_KEY=lsv2_pt_xxx
export LANGSMITH_PROJECT_SESSION_ID=<project-uuid>
```

### 步骤 2：创建告警规则

**方式 A — UI**：Project → **Alerts** → **Create Alert**，按下表配置。

**方式 B — API 脚本**：

```bash
chmod +x scripts/langsmith_alerts.example.sh
./scripts/langsmith_alerts.example.sh
```

规则定义见 [`scripts/langsmith_alerts.example.json`](../scripts/langsmith_alerts.example.json)。

### 步骤 3：rag-prod 推荐规则

| 名称 | Metric | 聚合 | 窗口 | 阈值 | Filter |
|------|--------|------|------|------|--------|
| HighErrorRate | Errors | pct | 15 min | ≥ 5% | tag `env:prod` |
| HighLatency | Latency | avg | 15 min | ≥ 30s | tag `rag`, name `rag_chat` |
| NegativeFeedbackSpike | Feedback Score | avg | 15 min | ≤ 0.3 | key `user_thumbs_down` |
| LowFaithfulness（A3 完成后） | Feedback Score | avg | 15 min | ≤ 0.6 | eval `rag_faithfulness` |

### 步骤 4：rag-dev 轻量规则

| 名称 | Metric | 聚合 | 窗口 | 阈值 |
|------|--------|------|------|------|
| DevErrors | Errors | pct | 15 min | ≥ 10% |
| DevLatency | Latency | avg | 15 min | ≥ 60s |

### 步骤 5：Slack Webhook

1. Slack → **Workflow Builder** → 新建 Workflow → **Webhook** trigger
2. 定义变量：`alert_rule_name`, `project_name`, `triggered_metric_value`, `triggered_threshold`, `runs_url`, `timestamp`
3. 添加 **Send a message** 到目标频道，正文引用上述变量
4. 复制 Webhook URL
5. LangSmith → 每条 Alert → **Notification** → **Webhook** → 粘贴 URL
6. 使用 **Preview** 验证 Slack 收到消息且 `runs_url` 可打开

### 验收

- [ ] `rag-prod` 三条核心告警已启用
- [ ] Slack 测试通知成功
- [ ] Preview 历史曲线与阈值合理

---

## Phase A — A2：Annotation Queue

**目标**：人工标注生产 trace，积累 Golden Dataset 原料（Phase C）。

### 创建 Queue

1. LangSmith → `rag-prod` → **Annotation Queues** → **Create**
2. 名称：`rag-prod-review`

### 标注 Schema

| 字段 | 类型 | 说明 |
|------|------|------|
| `router_correct` | bool | skip/retrieve 决策是否正确 |
| `answer_grounded` | bool | 答案是否有文档支撑 |
| `tool_appropriate` | bool | 工具调用是否必要 |
| `notes` | text | 自由备注 |

### 抽样优先级（手动 Add to queue）

1. `metadata.grounding_status` ∈ `not_supported`, `partial`
2. Feedback key = `user_thumbs_down`（尤其 `extra.reason=hallucination`）
3. Online Eval `rag_faithfulness` ≤ 0.5（A3 完成后）
4. `retrieval_plan.action=skip` 但用户问题明显需要文档

**每周目标**：10–20 条。

### 标注 SOP

1. 打开 trace → 查看 **Metadata**：`retrieval_plan`、`chunk_refs`、`grounding_status`
2. 阅读 Inputs（用户问题）与 Outputs（助手答案）
3. 若有 **Tools** 子 span，检查工具是否必要、结果是否被正确使用
4. 填写四项 schema；`notes` 记录可改进点（如「应 decompose」）
5. **Submit** 后可在 LangSmith 导出为 dataset（Phase C 使用）

### 验收

- [ ] Queue `rag-prod-review` 已创建
- [ ] 手动加入 ≥3 条 run 并完成标注
- [ ] 团队已阅读本 SOP

---

## Phase A — A3：Online Evaluators

**目标**：生产 trace 自动获得 faithfulness / relevance 分数。

### 启用

LangSmith → `rag-prod` → **Evaluators** → **Add Evaluator** → **LLM-as-Judge**

**采样建议**：上线前 2 周 100%；之后降至 20% 以控制成本。

**限制说明**：当前 metadata 含 `chunk_refs` 但不含 chunk 正文；faithfulness eval 主要依据 trace 内 model 输入上下文 + metadata。Phase B 将补充 `chunk_snippets` 摘要以提升精度。

### Evaluator 1：`rag_faithfulness`

**输入变量**：`inputs`（用户问题）、`outputs`（助手答案）、run metadata `chunk_refs`、`retrieval_plan.standalone_query`

**Prompt 模板**：

```
你是 RAG 质量评估员。根据检索上下文与用户问题，判断助手答案中的事实是否被检索材料支撑。

用户问题：
{question}

检索 query（改写后）：
{standalone_query}

检索到的 chunk 引用（document_id#chunk_index）：
{chunk_refs}

助手答案：
{answer}

评分规则：
- 1.0：答案核心事实均可从检索材料推断，无编造
- 0.5：部分事实有支撑，部分无法验证或可能编造
- 0.0：答案与检索材料无关或明显编造

只输出 0 到 1 之间的一个数字。
```

### Evaluator 2：`rag_answer_relevance`

**Prompt 模板**：

```
你是 RAG 质量评估员。判断助手答案是否直接回答了用户问题（答非所问则低分）。

用户问题：
{question}

助手答案：
{answer}

评分：1.0 完全相关，0.5 部分相关，0.0 答非所问或拒绝回答无理由。
只输出 0 到 1 之间的一个数字。
```

### Evaluator 3：`rag_retrieval_relevance`

**Prompt 模板**：

```
你是 RAG 检索评估员。根据用户问题与召回 chunk 引用列表，判断检索结果是否与问题相关。

用户问题：
{question}

召回 chunk 引用（含 filename）：
{chunk_refs}

检索策略：action={action}, strategy={strategy}

评分：1.0 召回高度相关，0.5 部分相关，0.0 完全不相关或 action=skip 但问题需要文档。
只输出 0 到 1 之间的一个数字。
```

### 与 GroundingMiddleware 的关系

| | GroundingMiddleware | Online Eval |
|--|---------------------|-------------|
| 时机 | turn 内同步 | LangSmith 异步 |
| 用途 | 前端展示、metadata | Dashboard、Alert、抽样 |
| 保留 | 是 | 新增 |

### 验收

- [ ] 三个 Evaluator 已在 `rag-prod` 启用
- [ ] 发送一条 chat 后，trace 上出现 eval 分数（约 5 分钟内）

---

## Phase A — A5：Dashboard 配置清单

LangSmith → `rag-prod` → **Monitoring** → **Create Dashboard**

| # | 面板名称 | 类型 | Filter / 维度 | 用途 |
|---|----------|------|---------------|------|
| 1 | Turn Volume | Run count | tag `rag` | 流量 |
| 2 | Error Rate | Error pct | tag `env:prod` | 稳定性 |
| 3 | Chat Latency | Latency P50/P99 | name `rag_chat` | 性能 |
| 4 | Grounding 分布 | Breakdown | metadata `grounding_status` | 生成质量 |
| 5 | Router 策略 | Breakdown | metadata `retrieval_plan.strategy` | 检索策略占比 |
| 6 | User Feedback | Feedback rate | keys `user_thumbs_up`, `user_thumbs_down` | 用户满意度 |
| 7 | Eval Faithfulness | Avg score | evaluator `rag_faithfulness` | 自动质量（A3 后） |

### 验收

- [ ] Dashboard 已创建且 7 个面板有数据（需 prod 流量）
- [ ] A4 上线后 Feedback 面板可见 👍/👎 比例

---

## Alert Playbook（速查）

见上文 A1。API 脚本：[`scripts/langsmith_alerts.example.sh`](../scripts/langsmith_alerts.example.sh)

Webhook 绑定必须在 **LangSmith UI 逐条 Alert 配置**（脚本只创建规则，不含 notification target）。

---

## 代码入口

| 模块 | 职责 |
|------|------|
| `app/observability/langsmith.py` | 启动时写 tracing 环境变量 |
| `app/observability/turn_trace.py` | turn 级 metadata 采集 |
| `app/observability/run_context.py` | 捕获 root run_id |
| `app/observability/langsmith_client.py` | patch metadata / submit feedback |
| `app/services/rag_service.py` | RunnableConfig tags + SSE `trace` 事件 |
| `app/routers/feedback.py` | 结构化 Feedback API |
| `app/schemas/feedback.py` | `FeedbackKind` / `FeedbackReason` |
