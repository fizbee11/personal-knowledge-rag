# Security Rules for config.cfg

**CRITICAL: `config.cfg` is a SECRETS FILE**

## Never include in commits
- `config.cfg` must never be staged, committed, or pushed to any repository
- Add `config.cfg` to `.gitignore` immediately if not already ignored

## Never expose verbatim content in other files
- Do NOT copy values from `config.cfg` into README.md, documentation, or code
- Replace specific URLs/paths with placeholders like `<YOUR_LLM_SERVER_URL>` or `<YOUR_QDRANT_URL>`
- Use environment variables or separate config system instead of hardcoded secrets

## Files that must NEVER contain config.cfg values:
- README.md (documentation)
- Any example files in the repo
- Dockerfile examples showing full config paths
- Commit messages or PR descriptions referencing specific internal URLs/paths

## Safe alternatives:
1. Use `<YOUR_...>` placeholders for documentation
2. Reference `config.cfg` as a separate file to configure at runtime
3. Store sensitive values in environment variables (e.g., `OLLAMA_BASE_URL`, `QDRANT_URL`)
4. Use `.env` files that are gitignored
