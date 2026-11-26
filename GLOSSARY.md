# rea-video

Reverse-engineering videos into shot-level breakdowns and assembling remakes with local tools, gateway APIs, and FFmpeg.

## Language

**Generation Unit**:
A contiguous or grouped set of shots packed into one video-model call (or split into timed parts when exceeding a model's duration cap).
_Avoid_: Job, batch, scene group

**Redrawn Frame**:
A still keyframe recreated or subject-swapped for a shot and stored under `remake/redrawn/`, overriding the raw source frame during video generation.
_Avoid_: Swapped still, edited reference, custom keyframe

**Bridge Stage**:
One independently resumable execution phase (`images`, `video`, `audio`, or `assemble`) of the local remake pipeline.
_Avoid_: Step, pass, runner phase

**Dubbing Script**:
The editable timeline of spoken lines and per-speaker voice bindings used to synthesize replacement narration before final assembly.
_Avoid_: Transcript, subtitle cues, narration dump

**Enrolled Voice**:
A cloned voice registered on the TTS server from a short reference audio clip under a unique non-built-in voice name.
_Avoid_: Cloned model, voice profile, speaker preset
