---
status: accepted
---

# Local Remake Bridge Without Capsule Cinema

Instead of requiring the external `capsule-cinema` repository or embedding heavy PyTorch TTS dependencies inside `rea-video`, this fork executes remakes via a native `rea-video bridge` command that calls a gateway supporting both OpenAI-compatible `/v1/images/*` (`gpt-image-*`) and Gemini `/v1beta/models/{model}:generateContent` (`gemini-*-image*`) (or local `gptimg` / `flow image`), dispatches video generation to local `flow` / `vids` CLIs, talks to `VieNeu-TTS` over its isolated HTTP OpenAI-compatible server, and feeds results into `rea-video`'s built-in `assemble()`. Gateway and VLM settings are loaded via `--env-file` or persisted with `rea-video prefs env_file`.

## Consequences

- Keeps `rea-video`'s Python environment lightweight and free of conflicting GPU/Torch dependencies.
- Enables safe `--dry-run` previews and human review checkpoints between stages (such as inspecting Redrawn Frames before spending video credits).
- Requires the user to run local supporting services when executing those stages (ShardBrowser for `flow` / `vids`, and the `VieNeu-TTS` HTTP server for dubbing).
