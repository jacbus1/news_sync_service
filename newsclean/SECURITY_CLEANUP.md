# Security Cleanup Report

This repository was prepared for public GitHub publication with a research-only scope.

## Removed from public copy
- Environment files (`.env`, `.env.*`, `*.env`)
- AWS-specific document (`AWS_OR_updated20260209.md`)
- Runtime artifacts (`*.log`, `*.local.log`, `*.db`, `src.zip`)
- Virtual environment and caches (`.venv`, `__pycache__`, `*.pyc`)
- Bot-oriented runtime scripts not required for research workflows

## Code-level hardening
- Removed absolute local filesystem defaults from retained scripts
- Removed `.env` loading behavior from retained entry workflows
- Updated CLI defaults to relative/local paths

## Re-check commands
Run these in repository root before pushing:

```bash
rg -n --hidden --glob '!.git' 'secret|token|api_key|[A-Za-z]:\\\\|^/.*'
rg --files -g '*.env' -g '.env*'
find . -type f \( -name '*.db' -o -name '*.log' -o -name '*.local.log' \)
```

Expected result: no matches for secrets/path scan, and no tracked secret/data runtime artifacts.
