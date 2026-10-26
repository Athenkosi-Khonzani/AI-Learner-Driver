# Implementation Status — Host v0.3.2

## Production integration status

The application now has the complete end-to-end software path required for the AI Learner Rover Driver:

| Component | Status | Implementation |
|---|---|---|
| PySide6 desktop UI | ✅ | `ui/` |
| Continuous camera capture | ✅ | `camera/` |
| Calibrated CV world state | ✅ | `perception/service.py` |
| Strong-tier LLM planning | ✅ | `planning/llm_providers.py` |
| Model selection | ✅ | Settings → Planner |
| Host-side plan validation | ✅ | `planning/validator.py` |
| Persistent WebSocket gateway | ✅ | `rover/gateway.py` |
| Full path/ACK/NACK/heading/telemetry protocol | ✅ | `rover/protocol.py` |
| Automatic telemetry enable on rover connection | ✅ | `application/mission_controller.py` |
| Post-path LLM goal verification | ✅ | `planning/verification.py` |
| Voice-to-text rover instructions | ✅ | `voice/service.py` |
| SQLite run/snapshot/telemetry history | ✅ | `persistence/sqlite_repository.py` |

## Strong-tier planner choices

The operator can select one of the three models evaluated in the strong-tier benchmark:

- **GPT-6 Sol** (`gpt-6-sol`) — default.
- **Claude Sonnet 5** (`claude-sonnet-5`).
- **Gemini 3.1 Pro** (`gemini-3.1-pro-preview`).

All providers receive the same canonical coordinate convention, rover-command semantics, measured rover footprint and host protocol limits. All responses are constrained with provider-native structured output and then passed through the same host-side `PlanValidator` before transmission.

Required environment variables:

```text
OPENAI_API_KEY
ANTHROPIC_API_KEY
GEMINI_API_KEY
```

Only the key for the selected planner provider is required for planning. Voice transcription uses OpenAI `gpt-transcribe`, so voice additionally requires `OPENAI_API_KEY`.

## Measured rover footprint

The production planner/world state now uses the measured top-down body dimensions:

```text
length/front-to-back = 260 mm (26 cm)
width/left-to-right  = 200 mm (20 cm)
```

The footprint is included in `scene_json["rover"]["footprint_mm"]` and explicitly repeated in the LLM system prompt so the planner reasons about the full body rather than a point robot.

## Protocol alignment

The bundled `docs/Command_Protocol_Specification.docx` is the source protocol used by this build.

### Laptop → Rover

- `path` — complete FWD/BACK/TURN/STOP plan with path id.
- `heading` — current filtered CV heading.
- `telem` — telemetry stream enable/disable.
- `speed` — debug base-speed setting.
- `pid` — debug PID tuning setting.

### Rover → Laptop

- `ack` — command or path completion.
- `nack` — command/path failure with reason.
- `telem` — `vbat`, `soc`, `dist` (`dist` is the rover ultrasonic distance measurement).
- `req_heading` — request latest CV heading.

The production UI and protocol parser use `dist` as the ultrasonic distance field. A legacy `cur` field is accepted only for backwards compatibility with earlier host/firmware revisions.

When `RoverConnectedEvent` is received and `telemetry_on_start` is enabled (default), the controller immediately queues:

```json
{"t":"telem","on":true}
```

## Voice workflow

Press **Mic** on Operator or Engineer:

1. the selected microphone records at 16 kHz mono;
2. capture ends after speech followed by silence, or the maximum recording duration;
3. OpenAI `gpt-transcribe` converts the recording to text;
4. the transcript is inserted into both instruction boxes;
5. the operator reviews it and presses **Send**.

Voice does not bypass the normal plan validation/confirmation/safety flow.

## Remaining engineering work

No intentional backend placeholders remain in the primary mission path. Remaining work is integration/field validation rather than missing architecture:

- exercise each provider from the real Settings model selector;
- test microphone thresholds in the actual lab noise environment;
- confirm/document the physical rover ultrasonic `dist` unit in the firmware/protocol;
- tune the post-path verification prompt/tolerances from physical runs;
- perform final end-to-end fault testing for Wi-Fi drop, camera drop, API failure and rover NACK.


### v0.3.4 camera-lag mitigation
- 10 s hard asynchronous wait after `req_heading`.
- Frame sequence baseline is captured only after the wait.
- Three post-delay frames must produce stable raw ArUco headings.
- Host heading timeout: 18 s.
- Recommended rover `HEADING_TIMEOUT_MS`: 30000 ms.
