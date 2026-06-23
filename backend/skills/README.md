# Skills 目录

本目录存放 Agent 可渐进式加载的领域技能（[Agent Skills 规范](https://docs.langchain.com/oss/python/deepagents/skills)）。

## 目录结构

```
skills/
  <skill-name>/
    SKILL.md          # 必需：YAML frontmatter + 指令正文
    scripts/          # 可选：可执行脚本
    references/       # 可选：参考文档
    assets/           # 可选：模板、静态资源
```

## SKILL.md 格式

```markdown
---
name: my-skill
description: 一句话说明何时激活此 skill（会出现在 Agent 启动时的技能目录中）
---

# 标题

具体指令、步骤、约束……
```

- `name`：唯一标识，供 `load_skill` 工具使用；省略时默认取目录名
- `description`：必填，Agent 据此判断是否需要加载

## 加载机制

1. **启动时**：`SkillsMiddleware` 仅将各 skill 的 `name` + `description` 注入 system prompt
2. **按需**：Agent 调用 `load_skill(skill_name)` 获取完整 `SKILL.md` 正文
3. **预热**：`build_agent()` 会调用 `load_all_skills()` 扫描并缓存

## 配置项（`backend/.env` 或环境变量）

| 变量 | 默认 | 说明 |
|------|------|------|
| `SKILLS_ENABLED` | `true` | 是否启用 SkillsMiddleware |
| `SKILLS_DIR` | `./skills` | 相对 `backend/` 根目录的技能路径 |
| `SKILLS_ALLOWLIST` | 空 | 逗号分隔的 skill 名称；空表示加载全部 |
| `SKILL_SCRIPT_ENABLED` | `true` | 是否启用 skill 脚本工具（`run_skill_script` 等） |
| `SKILL_SCRIPT_TIMEOUT_SECONDS` | `30` | 脚本执行默认超时（秒） |
| `SKILL_SCRIPT_MAX_TIMEOUT_SECONDS` | `300` | 单次执行超时上限 |
| `SKILL_SCRIPT_MAX_OUTPUT_BYTES` | `512000` | stdout/stderr 合并输出上限（约 500KB） |
| `SKILL_SCRIPT_MAX_FILE_BYTES` | `1048576` | 可读/可执行单文件大小上限（1MB） |
| `SKILL_SCRIPT_ALLOWED_EXTENSIONS` | `.py,.sh` | 允许执行的脚本扩展名 |

## 脚本执行与安全边界

Agent 可通过 `run_skill_script` 在 skill 的 `scripts/` 目录下执行 `.py` / `.sh` 脚本（需 HITL 批准）。

**编写规范：**

- 脚本放在 `<skill>/scripts/` 下，在 `SKILL.md` 中说明何时调用及参数含义
- 使用 `script_args` 列表传参，不使用 shell 管道或 `cd ..`
- 控制输出体积；过长输出会被截断
- 不要假设能访问 skill 目录外的文件或宿主 `.env` 密钥

**安全说明（本地受限 subprocess，非 OS 级 sandbox）：**

- 路径守卫：禁止 `..`、绝对路径；仅允许 skill 目录内白名单前缀
- 执行环境：`shell=False`，最小化 `env`，不继承应用密钥
- 脚本仍可执行任意 Python/Bash 逻辑（含联网）；仅适用于可信管理员维护的 skill
- 生产环境若需更强隔离，可后续替换为 Docker/远程 sandbox 实现

配套只读工具：`read_skill_file`（分页读 references/assets/scripts）、`list_skill_files`。

修改 `SKILL.md` 后需重建 Agent 才能刷新缓存：

```python
from app.agent.factory import rebuild_agent
rebuild_agent()
```

或在应用 lifespan 中暴露管理接口后调用。

## 添加新 skill

1. 在 `skills/` 下新建目录，例如 `skills/my-skill/SKILL.md`
2. 填写 frontmatter 与指令正文
3. 重启服务或调用 `rebuild_agent()`

## 官方 skill 库

LangChain 维护了 [langchain-skills](https://github.com/langchain-ai/langchain-skills) 仓库（13 个与 LangChain/LangGraph 相关的 skill）。可复制其中目录到本文件夹使用，例如：

```bash
# 示例：仅复制 RAG 相关 skill（需先 clone 仓库）
cp -r langchain-skills/skills/langchain-rag ./skills/
```

## 内置示例

- `example-rag-assistant`：本 RAG 项目的回答规范、检索配置说明与能力边界
