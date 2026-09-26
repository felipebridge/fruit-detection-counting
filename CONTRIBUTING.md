# Contributing

Thanks for your interest in improving Fruit Detection & Counting. Bug reports, fixes
and focused improvements are all welcome.

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/<your-username>/fruit-detection-counting.git
cd fruit-detection-counting
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Workflow

1. Open an issue first for anything larger than a small fix, so the approach can be agreed.
2. Branch from `main` with a short descriptive name, e.g. `fix/tracker-reset`.
3. Keep each pull request to one change, with tests for any behaviour you add or fix.
4. Write commit messages in the [Conventional Commits](https://www.conventionalcommits.org)
   style: `fix: ...`, `feat: ...`, `docs: ...`.

## Checks

Run these before opening a pull request. CI runs the same ones.

```bash
ruff check . && ruff format --check .   # lint and formatting
mypy                                    # type checks
pytest                                  # tests
```

One test runs real YOLO inference and is skipped unless `models/yolo11n.pt` exists.
Running the CLI once downloads it.

## Guidelines

- Keep the project focused on fruit detection and counting.
- Match the existing style: type hints, small functions, errors raised as `FruitCounterError`.
- New settings go in `configs/default.yaml` with a comment explaining them.
- Don't commit model weights, videos or generated outputs.

## Reporting issues

Use the [issue tracker](https://github.com/felipebridge/fruit-detection-counting/issues).
For bugs, include the command you ran, the full error output, your OS and Python version,
and, if you can, the input that triggers it.

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
