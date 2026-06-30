#!/usr/bin/env bash
# Bootstrap LangSmith alert rules via API (one-time manual setup).
#
# Prerequisites:
#   export LANGSMITH_API_KEY=lsv2_pt_xxx
#   export LANGSMITH_PROJECT_SESSION_ID=<project-uuid-from-langsmith-ui>
#
# Usage:
#   ./scripts/langsmith_alerts.example.sh

set -euo pipefail

: "${LANGSMITH_API_KEY:?Set LANGSMITH_API_KEY}"
: "${LANGSMITH_PROJECT_SESSION_ID:?Set LANGSMITH_PROJECT_SESSION_ID (project UUID)}"

ENDPOINT="${LANGSMITH_ENDPOINT:-https://api.smith.langchain.com}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

create_rule() {
  local payload="$1"
  curl -sf -X POST \
    -H "Authorization: Bearer ${LANGSMITH_API_KEY}" \
    -H "Content-Type: application/json" \
    "${ENDPOINT}/v1/platform/alerts/${LANGSMITH_PROJECT_SESSION_ID}" \
    -d "${payload}"
  echo
}

while IFS= read -r rule; do
  echo "Creating alert: $(echo "${rule}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["name"])')"
  create_rule "${rule}"
done < <(python3 -c "
import json
with open('${SCRIPT_DIR}/langsmith_alerts.example.json') as f:
    for rule in json.load(f):
        print(json.dumps(rule))
")

echo "Done. Next steps (LangSmith UI, per alert rule):"
echo "  1. Project -> Alerts -> open each rule -> Notification"
echo "  2. Add Webhook URL from Slack Workflow Builder"
echo "     See: https://support.langchain.com/articles/9581596180"
echo "  3. Map variables: alert_rule_name, triggered_metric_value, runs_url"
echo "  4. Or attach PagerDuty for production on-call"
