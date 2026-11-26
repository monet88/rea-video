# 本地桥接（rea-video bridge）

不依赖 `capsule-cinema`，直接在本机调度网关生图 / `gptimg` / `flow`、`flow video` / `vids`、`VieNeu-TTS` 配音与本地 FFmpeg 拼装成片。

## 能做什么

- **Images 阶段**（`--stage images`）：按 `production_plan.image_tasks` 重绘首帧 / 尾帧到 `remake/redrawn/`；支持 `gateway`（OpenAI `/v1/images/*` 如 `gpt-image-2.6-sunburst`，以及 Gemini `:generateContent` 如 `gemini-3.1-flash-image`）、`gptimg`、`flow`。
- **Video 阶段**（`--stage video`）：按生成单元自动路由 `flow video`（`edit` / `remix` / `keyframes` / `frames` / `text`）或 `vids`（`avatar` 镜头自动走 `vids`，也可传 `--video-backend vids`），输出 `remake/bridge/video/*.mp4` 并增量写入 `remake/mine.json`（含 `media_id` / `profile`）。支持 `--upscale 720p|1080p|4k`：可直接生成时超分，也可先 `--resolution 360p` 出草稿后再带 `--upscale 1080p` 对已有 `media_id` 零积分（≤10s）升级。
- **Audio 阶段**（`--stage audio`）：导出 `remake/dubbing_script.json`，可选 `--clone-source-voices` 从分离人声截取 5 秒样本向本地 `VieNeu-TTS`（默认 `http://127.0.0.1:8000` 或 `$REA_TTS_BASE_URL`）注册音色、合成配音并混音为 `remake/bridge/audio/dubbed_mix.wav`。
- **Assemble 阶段**（`--stage assemble`）：调用内置 `remake.assemble` 拼回 `remake/remake.mp4` 和 `remake/compare.mp4`（支持 `--only 1` / `--only G1` 只拼装指定镜头子集），若存在 `dubbed_mix.wav` 且未传 `--keep-original-audio` 则自动挂载配音音轨。

## 安全与环境配置

- **默认 `--dry-run`**：不加 `--run` 时只打印计划调用与跳过状态，不触发外部接口或生成；向用户确认后再加 `--run`。
- **网关生图环境变量**（可用 `rea-video prefs env_file <路径>` 持久化）：
  - `REA_IMAGE_API_BASE` / `REA_IMAGE_API_KEY`（未设时回退 `REA_GEMINI_API_BASE` / `REA_GEMINI_API_KEY`）
  - `REA_IMAGE_MODEL`（默认 `gpt-image-2.6-sunburst`；传 `gemini-*-image*` 自动走 `/v1beta/models/{model}:generateContent`）
- **配音服务**：`REA_TTS_BASE_URL`（默认 `http://127.0.0.1:8000`）。

## 常用顺序

```bash
rea-video plan out/ref --model gemini-omni-flash
rea-video bridge out/ref --dry-run
rea-video bridge out/ref --stage images --image-backend gateway --run
rea-video bridge out/ref --stage video --video-backend flow --run
rea-video bridge out/ref --stage audio --clone-source-voices --run
rea-video bridge out/ref --stage assemble --run
```
