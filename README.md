# AI Learner Rover Driver

**Final-year Electrical and Computer Engineering project — University of Cape Town**

A language-guided mobile robot system that turns natural-language instructions into rover actions using **overhead computer vision**, **large language model (LLM) planning**, and **feedback from the physical rover**.

This repository is the **software and experimentation hub** for the AI Learner Rover Driver. It contains the production desktop host and the Rover Driving Test (RDT) experiment harness. The independent LLM planning benchmark is maintained in a separate repository, linked below.

## Project overview

The goal is to let an operator describe a destination or task in ordinary language rather than manually specifying motor movements. The system identifies the rover and coloured objects in a controlled arena, builds a metric world-state description, asks an LLM for a structured motion plan, and sends validated commands to the rover. Following execution, a new camera observation is used to verify the result and, when necessary, request a bounded correction.

```text
             Natural-language instruction
                    (text / voice)
                            |
                            v
   Overhead camera --> Computer vision / world state
                            |
                            v
                  LLM high-level planner
                            |
                            v
               Host plan validation & UI
                            |
                            v
                  WebSocket connection
                            |
                            v
                   ESP32 / rover control
                            |
                            v
                    Physical execution
                            |
                            v
                Updated camera observation
                            |
                            v
                 Goal verification / retry
```

The **LLM chooses the high-level actions**, while the host validates plans and the rover-side control and safety logic handles physical execution. The system is designed around a controlled arena with coloured blocks and an overhead camera, rather than general outdoor navigation.

## Repository contents

| Component | Location | Purpose |
| --- | --- | --- |
| **AI Learner Host** | [`ai_learner_host/`](./ai_learner_host/) | Cross-platform desktop application for perception, planning, rover communication, operation, and run history. |
| **Rover Driving Test (Experiment 2)** | [`rdt_experiment/`](./rdt_experiment/) | Runs rover trials, records execution and camera evidence, replays runs, and generates analysis outputs. |
| **LLM Planning Tests (Experiment 1)** | **[Separate LLM Tests repository](https://github.com/Athenkosi-Khonzani/AI-Learner-Driver-LLM-Tests)** | Standalone simulated benchmark of LLM providers, structured plans, task success, latency, and cost. |

> **Note on `llm_tests/`:** An `llm_tests` entry may appear in this repository without displaying its source files. The maintained and accessible benchmark is hosted separately at **[AI-Learner-Driver-LLM-Tests](https://github.com/Athenkosi-Khonzani/AI-Learner-Driver-LLM-Tests)**. Use that repository for Experiment 1 source code, installation instructions, datasets, and benchmark scripts.

### 1. AI Learner Host

The host is the main operator-facing application. It brings together:

- **Perception:** Camera calibration, arena homography, ArUco-based rover localisation, and HSV detection of red, green, yellow, and blue blocks.
- **World-state generation:** Rover pose, heading, object positions, and arena geometry expressed in metric units for the planner.
- **LLM planning:** Configurable OpenAI, Anthropic, and Google planning providers with structured command output.
- **Execution:** WebSocket communication with the ESP32 gateway, command acknowledgements, telemetry, and camera-assisted heading responses.
- **Verification:** Post-execution observation, LLM-based goal checking, and bounded correction when needed.
- **Operator interface:** Manual/engineer controls, calibration settings, text and voice instructions, and local mission history.

For installation, configuration, protocol details, and tests, see the **[Host README](./ai_learner_host/README.md)**.

### 2. LLM Planning Tests — Experiment 1

**Repository:** [AI-Learner-Driver-LLM-Tests](https://github.com/Athenkosi-Khonzani/AI-Learner-Driver-LLM-Tests)

Experiment 1 evaluates the **high-level planning component independently of the physical rover** using a repeatable simulator and a fixed set of scenarios. The benchmark includes provider adapters, structured-output checks, simulated execution and collision evaluation, repeated trials, result visualisation, and replay verification.

The standalone test repository contains the scenario datasets, configurations, benchmark runners, results, and instructions for the different experimental phases. **It does not modify or import the production host application.**

### 3. Rover Driving Test — Experiment 2

The [`rdt_experiment/`](./rdt_experiment/) harness evaluates the **integrated rover system** rather than the planner alone. It uses the production host and Experiment 1 simulator components to run and assess tasks from controlled initial states.

It supports simulated rehearsals and physical trials, trial videos and annotated frames, pose trails and protocol logs, reproducible scoring, and generated figures/tables for analysis. Large ZIP archives of experimental results are intentionally not stored in this Git repository.

For the full procedure, commands, and output formats, see the **[RDT README](./rdt_experiment/README.md)** and its [`docs/OPERATOR_GUIDE.md`](./rdt_experiment/docs/OPERATOR_GUIDE.md).

## Getting started

### Requirements

- **Python 3.11+** for the desktop host.
- A supported **Windows or Linux** environment.
- Relevant API credentials for the selected LLM provider (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and/or `GEMINI_API_KEY`).
- For physical operation: a configured rover, its ESP32 WebSocket connection, an overhead camera stream, and saved camera/arena calibration.

### Install and launch the host

Clone the repository:

```bash
git clone git@github.com:Athenkosi-Khonzani/AI-Learner-Driver.git
cd AI-Learner-Driver/ai_learner_host
```

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python run.py
```

**Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
python run.py
```

In the host settings, configure the rover connection and camera, load the intrinsic calibration, and complete the arena and colour calibration before attempting a physical mission. Refer to the Host README for details.

### Run the Rover Driving Test harness

Install the separate [LLM Tests repository](https://github.com/Athenkosi-Khonzani/AI-Learner-Driver-LLM-Tests) as a sibling folder named `llm_tests`, alongside `ai_learner_host` and `rdt_experiment`, as described in the RDT documentation.

From the repository root:

```bash
cd rdt_experiment
pip install -r requirements.txt
python scripts/run_rdt.py --simulate
```

`--simulate` rehearses the experiment without physical hardware or LLM API costs. For the physical experiment, the host must first be configured and calibrated; follow the [RDT operator guide](./rdt_experiment/docs/OPERATOR_GUIDE.md) before running `python scripts/run_rdt.py`.

## Experiments at a glance

| | Experiment 1: LLM Planning | Experiment 2: Rover Driving Test |
| --- | --- | --- |
| **Primary question** | Can an LLM produce valid, effective rover plans from a structured world state? | How well does the complete perception–planning–execution system perform on physical driving tasks? |
| **Environment** | Deterministic simulation | Physical arena, with a simulated rehearsal mode |
| **Primary evidence** | Planning success, command validity, latency, cost, replays | Task outcomes, rover trajectories, timing, video, logs, scoring |
| **Code** | [Separate benchmark repository](https://github.com/Athenkosi-Khonzani/AI-Learner-Driver-LLM-Tests) | [`rdt_experiment/`](./rdt_experiment/) |

## Other project repositories

This repository serves as the central hub for the desktop host and experimental tools, but other parts of the AI Learner Rover Driver are maintained in separate GitHub repositories:

| Repository | Contents |
| --- | --- |
| **[AI Learner CV Pipeline](https://github.com/Athenkosi-Khonzani/KHNATH002-CV-Tests)** | Source code for the overhead computer-vision pipeline, including camera calibration, rover localisation, and coloured-block detection. The marker-height correction developed for the integrated evaluation is included under Experiment 2 in this repository. |
| **[AI Learner Rover Firmware](https://github.com/Athenkosi-Khonzani/AI-Rover-Firmware-Test)** | Embedded rover software, including STM32 motion control and its UART interface, plus the ESP32 communication gateway. |
| **[AI Learner PCB Shield](https://github.com/Athenkosi-Khonzani/Acebott_Shield_Version_A)** | KiCad design project and fabrication files for the Design 1 motor-control shield. |

The **[LLM Planning Tests (Experiment 1)](https://github.com/Athenkosi-Khonzani/AI-Learner-Driver-LLM-Tests)** are also maintained separately, as described above.

Large raw recordings and experimental ZIP archives are intentionally excluded from this repository. For implementation details or build instructions, consult the README in the relevant component repository.

## Author

**Athenkosi Khonzani**  
Final-year Electrical and Computer Engineering project  
University of Cape Town (UCT)

---

*For component-specific setup and implementation details, start with the [Host README](./ai_learner_host/README.md), the [RDT README](./rdt_experiment/README.md), or the [LLM Tests repository](https://github.com/Athenkosi-Khonzani/AI-Learner-Driver-LLM-Tests).*
