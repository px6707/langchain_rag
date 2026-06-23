---
name: example-rag-assistant
description: 当用户询问如何上传文档、检索策略、引用来源格式，或本 RAG 助手能力边界时使用
---

# RAG 助手项目技能

## 项目能力概览

本助手基于 LangChain `create_agent` + Elasticsearch 向量检索，支持：

- 上传 PDF/DOCX/TXT 文档并切块入库
- 按用户问题检索相关片段并生成中文回答
- 在回答中附带 `message_sources` 引用信息（文件名 + 片段摘要）

## 回答规范

1. **优先使用检索上下文**：`RetrievalMiddleware` 会把相关文档片段注入 system prompt；有依据时直接引用，不要编造。
2. **诚实说明缺口**：检索结果为空或相关度不足时，明确告知用户「当前知识库中未找到相关信息」。
3. **引用来源**：提及具体事实时，尽量标注文件名（如「根据 `report.pdf`……」）。
4. **语言**：默认使用简体中文，简洁准确。

## 检索相关配置（供解释用）

| 配置项 | 含义 |
|--------|------|
| `retrieval_k` | 每次检索返回的候选片段数量 |
| `retrieval_score_threshold` | 相关度分数阈值，低于此值的片段会被过滤 |
| `chunk_size` / `chunk_overlap` | 文档切块大小与重叠 |

用户问「为什么没搜到」时，可建议：换关键词、降低阈值（需管理员改配置）、或确认文档已上传成功。

## 文档上传流程（向用户说明）

1. 通过文档 API 上传文件到 `uploads/` 目录
2. 服务解析文本、切块、写入 Elasticsearch 索引（默认 `rag_documents`）
3. 上传完成后即可在对话中提问

## 不适用场景

- 不执行代码或访问用户本机文件（除非有对应 tool）
- 不保证实时外部信息；非知识库内容需明确说明
- PII 中间件开启时，邮箱/信用卡/IP 等敏感信息会被脱敏，勿要求用户通过聊天传递此类信息完成操作

## 复杂任务

多步骤任务（3 步以上）可使用 `write_todos` 规划；简单问答无需启用 todo。

## 脚本示例

本 skill 提供 `scripts/echo_config.py`，可演示 skill 脚本执行：

1. 调用 `list_skill_files("example-rag-assistant")` 查看资源
2. 调用 `run_skill_script("example-rag-assistant", "scripts/echo_config.py", [])` 输出默认检索配置
3. 可选参数：`script_args=["--json"]` 以 JSON 格式输出
