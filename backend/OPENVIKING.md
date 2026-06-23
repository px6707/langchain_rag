# OpenViking 集成说明

本项目使用 **ES 文档 RAG + LangGraph Checkpointer + OpenViking（含偏好）** 组合，**不接入 Honcho**。

## 职责分工

| 组件 | 职责 |
|------|------|
| **Elasticsearch** | 上传文档向量检索、来源引用 |
| **LangGraph Checkpointer** | 会话运行态、HITL、todos、history API |
| **OpenViking** | 对话归档、跨 session 长期记忆（含 preferences / profile / entities / events） |

## 启用步骤

1. 安装依赖：`pip install -r requirements.txt`（含 `openviking`）

2. 配置 `ov.conf`（VLM 用于 commit 后记忆抽取，embedding 用于语义检索）：

   ```bash
   mkdir -p ~/.openviking
   cp backend/openviking.ov.conf.example ~/.openviking/ov.conf
   # 编辑 api_key / model 等
   ```

   或使用官方向导：

   ```bash
   openviking-server init
   openviking-server doctor
   ```

3. 在 `.env` 中启用：

   ```env
   OPENVIKING_ENABLED=true
   OPENVIKING_PATH=./openviking_data
   # OPENVIKING_CONFIG_FILE=~/.openviking/ov.conf
   ```

4. 启动 backend，检查 `/health` 中的 `openviking_available` / `openviking_healthy`。

## 数据流

**写入（每轮对话）**

- JWT `user.id` → OpenViking `user_id`
- 前端 `session_id` → OV Session ID（与 LangGraph `thread_id` 一致）
- 每轮 user/assistant 消息写入 OV Session
- 达到 `OPENVIKING_COMMIT_EVERY_MESSAGES`（默认 20）时自动 `commit()`，异步抽取长期记忆（含偏好）

**读取（每轮模型调用前）**

- `OpenVikingMemoryMiddleware` 对 `viking://user/{user_id}/memories/` 做语义检索
- 将 profile / preferences / entities / events 等注入 system prompt
- 随后 `RetrievalMiddleware` 再注入 ES 文档片段

## 配置项

| 环境变量 | 说明 | 默认 |
|----------|------|------|
| `OPENVIKING_ENABLED` | 总开关 | `false` |
| `OPENVIKING_PATH` | 嵌入式数据目录 | `./openviking_data` |
| `OPENVIKING_CONFIG_FILE` | 覆盖 ov.conf 路径 | 空（用 `~/.openviking/ov.conf`） |
| `OPENVIKING_ACCOUNT_ID` | 多租户 account | `default` |
| `OPENVIKING_FIND_LIMIT` | 记忆检索条数 | `5` |
| `OPENVIKING_FIND_SCORE_THRESHOLD` | 记忆相关度阈值 | `0.3` |
| `OPENVIKING_COMMIT_EVERY_MESSAGES` | 触发 commit 的消息数 | `20` |

## 注意事项

- OpenViking 初始化失败不会阻断主服务；`/health` 会暴露 `openviking_error`。
- `commit()` 依赖 VLM，失败只记日志，不影响聊天。
- 文档 RAG 仍走 ES，上传文档不会自动进入 OpenViking Resources。
