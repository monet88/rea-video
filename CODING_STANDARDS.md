# Coding Standards

Judgement-based review rules for `rea-video` (checked during code review). Mechanical lint, test, and schema checks run via `ruff`, `pytest`, and `scripts/validate_capabilities.py`.

- **CLI bilingual parity**: Every user-facing subcommand, flag, and `--help` string added or changed in `rea_video/cli.py` must be mirrored in `rea_video/cli_en.py`.
- **Preview & backend parity**: `--dry-run` output must mirror `--run` path/skip logic without external calls; shared flags and reference inputs (`ref_paths`, `--audio`, `--only`) must behave consistently across backends.
- **No silent flag drops**: If an option combination is unsupported on a branch (for example `--retime warp --audio`), fail fast with an explicit error instead of ignoring the flag.
- **Reuse repo helpers**: Prefer existing utilities in `rea_video/media.py` and `rea_video/util.py` over duplicating FFmpeg/HTTP wrappers, and use stdlib before adding runtime dependencies.
- **Windows CLI contracts**: Flatten embedded newlines in arguments passed to `.cmd` shims, and pass multi-value `nargs="*"`/`"+"` options as a single flag followed by all paths (`--ref a b`).
- **Secret hygiene**: Load API keys and gateway URLs from env vars, `--env-file`, or `prefs env_file`; never include secrets in cache keys, logs, tests, or commits.
