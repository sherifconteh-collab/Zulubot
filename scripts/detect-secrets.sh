#!/usr/bin/env bash
# Pre-commit hook: detect potential secrets in staged files
# Patterns cover common API keys, tokens, and credentials

set -euo pipefail

PATTERNS=(
  'AKIA[0-9A-Z]{16}'                          # AWS Access Key
  'AIza[0-9A-Za-z\-_]{35}'                    # Google API Key
  'ghp_[0-9A-Za-z]{36}'                       # GitHub Personal Access Token
  'gho_[0-9A-Za-z]{36}'                       # GitHub OAuth Token
  'github_pat_[0-9A-Za-z_]{82}'               # GitHub Fine-grained PAT
  'sk-[0-9A-Za-z]{48}'                        # OpenAI / Anthropic API Key
  'sk-ant-[0-9A-Za-z\-]{80,}'                 # Anthropic API Key (long form)
  'xoxb-[0-9]{10,}-[0-9A-Za-z]{24}'           # Slack Bot Token
  'xoxp-[0-9]{10,}-[0-9]{10,}-[0-9A-Za-z]+'  # Slack User Token
  'SG\.[0-9A-Za-z\-_]{22}\.[0-9A-Za-z\-_]{43}' # SendGrid API Key
)

FOUND=0

# Check staged files (or all tracked files if not in a git hook context)
if git rev-parse --git-dir > /dev/null 2>&1; then
  FILES=$(git diff --cached --name-only --diff-filter=ACMR 2>/dev/null || git ls-files)
else
  FILES=$(find . -type f -not -path './.git/*')
fi

for pattern in "${PATTERNS[@]}"; do
  MATCHES=$(echo "$FILES" | xargs grep -rEn "$pattern" 2>/dev/null \
    | grep -v '\.env\.example' \
    | grep -v 'detect-secrets\.sh' \
    | grep -v 'node_modules' \
    || true)
  if [ -n "$MATCHES" ]; then
    if [ "$FOUND" -eq 0 ]; then
      echo "🚨 Potential secrets detected! Review before committing:"
      echo ""
    fi
    echo "$MATCHES"
    echo ""
    FOUND=1
  fi
done

if [ "$FOUND" -eq 1 ]; then
  echo "If these are false positives, use: git commit --no-verify"
  exit 1
fi

exit 0
