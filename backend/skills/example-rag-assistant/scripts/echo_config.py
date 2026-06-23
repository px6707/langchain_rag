#!/usr/bin/env python3
"""示例：输出 RAG 检索相关默认配置（供 run_skill_script 演示）。"""

import json
import sys

DEFAULTS = {
    "retrieval_k": 4,
    "retrieval_score_threshold": 0.7,
    "chunk_size": 500,
    "chunk_overlap": 50,
}


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        print(json.dumps(DEFAULTS, ensure_ascii=False))
    else:
        for key, value in DEFAULTS.items():
            print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
