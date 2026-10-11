## Description

<!-- What does this PR do, and why? -->

Fixes #<!-- issue number, or remove this line -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Security fix
- [ ] Documentation
- [ ] Refactor (no behavior change)
- [ ] Tests
- [ ] CI / tooling

## Changes

-
-

## Testing

```bash
python -m unittest discover -s tests -p "test_*.py"
python scripts/security_checks.py
```

- [ ] New behavior has tests, with external APIs mocked
- [ ] Existing tests still pass

## Money and risk checklist

- [ ] `PAPER_MODE=true` is still the default and nothing here can place a real order without `PAPER_MODE=false`
- [ ] Orders are only booked when the exchange confirms the fill
- [ ] Risk limits (caps, drawdown halts, stake limits) are respected, not bypassed
- [ ] User-supplied amounts and prices are bounded and validated

## Security checklist

- [ ] No hardcoded secrets, API keys or personal URLs (tunnels, IPs)
- [ ] No `eval`/`exec`, `shell=True`, `pickle` or other dynamic code execution
- [ ] Any new Telegram handler calls `_check_auth` first
- [ ] Untrusted input (market data, screenshots, chat text) is validated and treated as data
- [ ] New dependencies are from trusted sources, version-bounded, and pass `pip-audit`
- [ ] New GitHub Actions are pinned to a commit SHA

## Licensing

- [ ] I have signed the [CLA](https://github.com/sherifconteh-collab/Zulubot/blob/main/CLA.md) (the bot will prompt me on my first PR)

## Docs

- [ ] README / `.env.example` / `CHANGELOG.md` updated if user-facing

## Notes

<!-- Anything reviewers should know -->
