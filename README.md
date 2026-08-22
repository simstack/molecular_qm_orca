# molecular_qm_orca

ORCA capabilities for molecular quantum mechanics within the Simstack framework.

## Layout

- `nodes/` — `orca` node (single-point and geometry optimization)
- `models/` — ORCA uses shared `QMInput` from `molecular_qm_models`
- `lib/` — input writer, output parser, result builders
- `testing/` — manual/integration scripts
- `tests/` — tests that do not require a committed `uv.lock`

## Host usage

Installed packages (`simstack`, `molecular_qm_models`) are registered via
`simstack.modules` entry points. After `uv sync`:

```bash
uv run create_model_table
uv run create_node_table
```

Standalone in this repo, `create_*_table --dir` only accepts directories
**inside this repo**:

```bash
uv sync
uv run create_model_table --dir molecular_qm_orca
uv run create_node_table --dir molecular_qm_orca
```

## Install from git

Parent projects (for example jia-project) should depend on this package from
git, not as a submodule:

```toml
molecular_qm_orca = { git = "https://github.com/simstack/molecular_qm_orca.git" }
```
