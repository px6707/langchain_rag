# MCP 集成说明

本项目通过 [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters) 将 MCP Server 的工具转为 LangChain `BaseTool`，与本地 `app/tools/` 工具一并注册到 Agent。

## 配置文件

默认读取 [`mcp_servers.json`](mcp_servers.json)（相对 `backend/` 根目录）。模板见 [`mcp_servers.example.json`](mcp_servers.example.json)。

### stdio 示例（本地子进程）

```json
{
  "math": {
    "transport": "stdio",
    "command": "python",
    "args": ["./examples/mcp_math_server.py"]
  }
}
```

`args` 中以 `./` 开头或以 `.py` 结尾的路径会解析为相对 `backend/` 的绝对路径；`python` 会自动替换为当前 venv 的解释器。

### HTTP 示例（远程 MCP Server）

```json
{
  "weather": {
    "transport": "http",
    "url": "http://localhost:8000/mcp"
  }
}
```

### HTTP 鉴权（`auth` 元数据）

远程 MCP（`http` / `sse`）可通过 `auth` 块声明鉴权方式。密钥只写在 `.env`，JSON 里仅引用环境变量名。传给 `langchain-mcp-adapters` 前，`auth` 会被剥离并转换为 `headers`。

**Bearer：**

```json
{
  "weather": {
    "transport": "http",
    "url": "https://mcp.example.com/mcp",
    "auth": {
      "type": "bearer",
      "token_env": "MCP_WEATHER_TOKEN"
    }
  }
}
```

`.env`：`MCP_WEATHER_TOKEN=your-token`

**API Key（自定义 header）：**

```json
{
  "partner": {
    "transport": "http",
    "url": "https://partner.example.com/mcp",
    "auth": {
      "type": "api_key",
      "header": "X-API-Key",
      "token_env": "MCP_PARTNER_API_KEY"
    }
  }
}
```

**Basic Auth：**

```json
{
  "legacy": {
    "transport": "sse",
    "url": "https://legacy.example.com/sse",
    "auth": {
      "type": "basic",
      "user_env": "MCP_LEGACY_USER",
      "pass_env": "MCP_LEGACY_PASS"
    }
  }
}
```

规则：

- 无 `auth` 块：与之前行为一致
- 已有 `headers`：与 `auth` 合并；同名 header 以 `auth` 为准
- env 缺失或为空：跳过该 server，不影响其他 server
- `stdio` / `websocket` 上的 `auth` 会被忽略并打 warning

与 stdio 的 `env` + `${VAR}` 不同：`auth` 用于 HTTP 请求头，不把密钥传给子进程。

完整示例见 [`mcp_servers.example.json`](mcp_servers.example.json)。

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `MCP_ENABLED` | `true` | 是否加载 MCP 工具 |
| `MCP_SERVERS_FILE` | `./mcp_servers.json` | Server 清单 JSON 路径 |
| `MCP_TOOL_ALLOWLIST` | 空 | 逗号分隔工具名；空表示加载全部 |

## 示例 Math Server

[`examples/mcp_math_server.py`](examples/mcp_math_server.py) 提供 `add`、`multiply` 两个工具，基于 FastMCP stdio 模式。

单独测试 server（可选）：

```bash
cd backend
./venv/bin/python examples/mcp_math_server.py
```

## 启动与热更新

1. 确保 `mcp_servers.json` 配置正确
2. 启动 FastAPI：`uvicorn app.main:app --reload --port 8000`
3. 访问 `GET /health` 查看 `mcp_tools_count`、`mcp_tool_names`
4. 访问 `GET /api/chat/tools` 查看 Agent 已注册工具列表

修改 `mcp_servers.json` 后需**重启服务**，或在代码中调用 `rebuild_agent()`（会执行 `reload_mcp` + `reload_tools`）。

## 工具命名

默认启用 `tool_name_prefix=True`，MCP 工具名格式为 `{server}_{tool}`，例如 `math_add`、`math_multiply`，避免多 server 工具名冲突。

本地工具与 MCP 工具同名时，**本地工具优先**。

## HITL

将 MCP 工具名加入 `.env` 的 `HITL_TOOLS`（逗号分隔），即可复用现有人工审批中间件。

## 限制

- 默认 **stateless** 模式：每次工具调用新建 MCP session，适合 Web 部署
- 有状态 MCP server 需后续按请求维持 session（未纳入当前实现）
- 仅加载 **tools**；MCP resources / prompts 暂未接入
