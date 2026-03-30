#!/usr/bin/env bash
# Install git hooks for the Zulubot project
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_DIR="$REPO_ROOT/.git/hooks"

# Pre-commit: secret detection
cat > "$HOOK_DIR/pre-commit" << 'HOOK'
#!/usr/bin/env bash
exec bash "$(git rev-parse --show-toplevel)/scripts/detect-secrets.sh"
HOOK
chmod +x "$HOOK_DIR/pre-commit"

echo "Git hooks installed successfully."
