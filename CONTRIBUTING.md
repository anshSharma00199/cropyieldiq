# Contributing

## Branching (GitHub Flow + a long-lived develop branch)
- `main`: always deployable; deploys to production through CD. Protected: PR required, 1 approval, CI green, no force-push.
- `develop`: integration branch (optional for a small team; you may branch straight from `main`).
- `feature/<short-name>`, `fix/<short-name>`, `docs/<short-name>`: one topic per branch, opened as a PR.

## Commits (Conventional Commits)
`feat: add sensitivity endpoint`, `fix: reject ph above 14`, `docs: explain rate limits`, `test:`, `chore:`, `ci:`.
Small, focused commits. Reference issues with `Closes #12`.

## Releases
Tag `vMAJOR.MINOR.PATCH` on main (`git tag v1.2.0 && git push --tags`). CD publishes the image with that tag.
Rollback = redeploy the previous tag.

## Local setup
```
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
make install && make test
pre-commit install
```

## Data and models
Raw data is tracked with DVC or git (check the Kaggle licence before publishing). Model artifacts are NOT committed:
CI and Docker rebuild them from `ml/train.py`; production can pull them from object storage.
