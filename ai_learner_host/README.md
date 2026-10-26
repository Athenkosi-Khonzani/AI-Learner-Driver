# AI Learner Rover Driver — Host Application v0.3.5

Cross-platform Python desktop host for the final-year **AI Learner Rover Driver**. The application now integrates calibrated overhead perception, selectable strong-tier LLM planning, persistent ESP32 WebSocket control, voice-to-text instructions, post-path verification and local experiment/run history.

## Production mission flow

```text
Text or voice instruction
        ↓
Calibrated overhead world state
        ↓
Selected strong-tier LLM planner
        ↓
Provider-native structured command output
        ↓
Host PlanValidator
        ↓
Optional operator approval
        ↓
One complete WebSocket path
        ↓
ESP32 ACK / NACK / req_heading / telemetry
        ↓
Post-path camera capture
        ↓
LLM goal verification
        ↓
Complete OR bounded correction path
```

The rover-side ultrasonic/safety layer remains independent of the slower high-level LLM planning loop.

## Planning models

Settings → **Planner** allows the operator to switch between:

| UI label | Provider | API model id |
|---|---|---|
| **GPT-6 Sol** | OpenAI | `gpt-6-sol` |
| Claude Sonnet 5 | Anthropic | `claude-sonnet-5` |
| Gemini 3.1 Pro | Google | `gemini-3.1-pro-preview` |

**GPT-6 Sol is the default** based on the project's strong-tier benchmark result. The application does not hard-code provider secrets. Set the relevant Windows/Linux environment variables:

```text
OPENAI_API_KEY
ANTHROPIC_API_KEY
GEMINI_API_KEY
```

All three provider adapters use the same structured command schema and the same host-side validator. Planning requests use medium reasoning/thinking, matching the strong-tier comparison condition.

## Measured rover size

The real rover footprint has now been measured and is used by both perception/world state and the planner prompt:

```text
260 mm long (front-to-back)
200 mm wide (left-to-right)
```

The LLM is explicitly told to plan for the **full rectangular footprint**, keep it inside the arena and avoid obstacles rather than treating the rover as a point.

## Voice-to-text

Voice input is live rather than a placeholder.

Press **Mic** on either Operator or Engineer. The host records from the selected microphone until speech ends, then sends the WAV recording to OpenAI's `gpt-transcribe`. The transcript is placed in the instruction box for review before you press **Send**.

Voice requires `OPENAI_API_KEY` even if Claude or Gemini is selected as the planner.

## Rover protocol

The build follows `docs/Command_Protocol_Specification.docx`.

### Host → Rover

```json
{"t":"path","id":7,"cmds":[["TURN",-30],["FWD",200],["STOP"]]}
{"t":"heading","deg":92.5}
{"t":"telem","on":true}
{"t":"speed","base":180}
{"t":"pid","kp":1.2,"ki":0.0,"kd":0.4}
```

### Rover → Host

```json
{"t":"ack","of":"cmd","id":7,"i":1}
{"t":"ack","of":"path","id":7}
{"t":"nack","id":7,"i":1,"reason":"obstacle"}
{"t":"telem","vbat":7.82,"soc":74,"dist":14}
{"t":"req_heading"}
```

Wire command ACK index `i` is treated as **1-based** and normalized internally to Python's 0-based indexing.

On every rover connection, telemetry is enabled automatically by default:

```json
{"t":"telem","on":true}
```

The production telemetry field is `dist`, representing the rover's ultrasonic distance measurement. The host still accepts legacy `cur` messages so older logs/firmware revisions do not crash the application.

## Installation

Python 3.11+ is required.

### Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python run.py
```

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
python run.py
```

Main dependencies now include:

```text
PySide6
qasync
websockets
opencv-contrib-python
numpy
pydantic
openai
anthropic
google-genai
sounddevice
```

On Linux, microphone capture may additionally require the distribution's PortAudio package if it is not already installed.

## Calibration / first launch

The existing CV calibration chain is unchanged:

1. Settings → Connection: configure rover WebSocket and camera stream.
2. Settings → Camera: load intrinsic calibration JSON.
3. Calibrate the arena homography.
4. Calibrate red/green/yellow/blue blocks.
5. Confirm `DICT_4X4_50`, marker ID 29 localisation and the stored marker-to-centre offset.

Your current calibration/settings are preserved in `data/settings.json`.

## Safety properties

- LLMs never write directly to the WebSocket.
- LLMs produce only FWD/BACK/TURN/STOP proposals.
- Path id and transport metadata are always host-owned.
- Every LLM plan passes the same numeric/operation validator.
- STOP is mandatory and terminal.
- A mission cannot be manually jogged concurrently.
- NACK triggers a fresh world-state capture and safe re-plan.
- `req_heading` is answered from filtered CV perception; no fake heading is sent if perception cannot provide one.
- Confirmation-before-send remains enabled by default.
- Voice transcription is reviewed as text before mission submission.
- The ESP32's local safety layer remains authoritative for immediate obstacle stopping.

## Project structure

```text
src/ai_learner_host/
├── app.py
├── config.py
├── application/
├── camera/
├── domain/
├── perception/
├── persistence/
├── planning/
│   ├── llm_providers.py
│   ├── schemas.py
│   ├── planner.py
│   ├── prompt_builder.py
│   ├── validator.py
│   └── verification.py
├── rover/
├── ui/
└── voice/
    └── service.py
```

## Development rover simulator

To test the WebSocket protocol without the physical rover:

```powershell
python -m tools.fake_rover --port 8765
```

Set the host Rover URI to:

```text
ws://127.0.0.1:8765
```

The fake rover supports telemetry, path ACKs and heading requests.

## Tests

```powershell
pytest -q
```

The test suite covers protocol encoding/decoding, telemetry-on-connect, perception helpers, plan validation, provider/model selection, the production prompt footprint, post-path correction parsing and WAV encoding for voice capture.


## v0.3.2 heading handshake fix

- Matches the physical ESP32 firmware's zero-based ACK/NACK command indexes.
- Persists camera-heading replies in the event log for turn debugging.
- Treats rover `{"t":"invalid"}` replies as explicit protocol errors.
- The corresponding rover firmware executes queued paths outside the WebSocket callback so `req_heading` / `heading` can complete without a re-entrant WebSocket deadlock.


## v0.3.3 fresh-heading gate

Camera-verified turns no longer answer `req_heading` from whatever frame happens
to be cached at that instant.

The host now:

1. timestamps the rover's `req_heading`;
2. waits 0.45 s for the continuously-drained camera stream to catch up;
3. ignores frames that existed before the request/settling boundary;
4. samples three distinct new frames;
5. obtains raw ArUco headings without the temporal heading filter;
6. requires those headings to agree within 6°;
7. only then sends `{"t":"heading","deg":...}`.

If a stable fresh heading is unavailable within 3.5 s, the host sends no stale
fallback heading. The rover's existing heading-timeout path remains authoritative.

The ESP32 heading timeout in the paired v0.3.2 firmware is already 50 seconds,
so increasing it is not required for this host-side freshness fix.


## v0.3.4 conservative 10-second camera settle

Physical testing showed that the overhead camera stream can remain visually stale even
while new frames are arriving at roughly 6 FPS. Heading verification therefore now uses
a hard 10-second asynchronous settle period after each rover `req_heading`. Only after
that full delay does the host snapshot the frame sequence and collect three newer raw
ArUco heading samples. The host heading acquisition timeout is 18 seconds.

Recommended paired ESP32 setting:

```cpp
static const unsigned long HEADING_TIMEOUT_MS = 30000;
```

The 30-second rover timeout deliberately exceeds the host's 18-second heading window.


## v0.3.5 settled final verification frame

Final goal verification now uses the same `heading_reply_settle_seconds` camera-lag
tuning value as camera-verified turns.  After a path ACK, the host enters VERIFYING,
waits for that settle interval, establishes a new frame-sequence baseline, then waits
for `heading_reply_samples` frames that arrive after the delay.  The newest of those
frames is used to rebuild the final world state supplied to the verification LLM.
The rover pose filter is reset before processing that final frame so the final pose is
not blended with the pre-path pose.
