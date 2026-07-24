# CoasterBench: end-to-end agentic tool-calling eval

[CoasterBench](https://github.com/tlrmchlsmth/CoasterBench) has a vLLM-served
model design RollerCoaster Tycoon 2 coasters through OpenAI tool calling: each
round the model submits a JSON track program (forced tool use), a headless
OpenRCT2 build validates and simulates it, and the model iterates on the eval
report. The game runs with zero RCT2 assets (`--no-graphics`), so everything
is redistributable and CPU-only; the only GPU cost is the model under test.

## What this covers that parser unit tests don't

A full multi-turn loop against a real server: `tool_choice: "required"` and
named-function forcing (guided decoding), tool_result round-trips, and
multi-KB structured JSON arguments (a 148-piece track program), all under a
~3k-token system prompt. On its first day against a live server this setup
surfaced two API-contract violations — `required` returning zero tool calls
(cold-start), and named `tool_choice` returning a *different* function than
the one forced (`poolside_v1` parser) — plus a harness-side lesson about
reasoning models exhausting fixed completion budgets before their first tool
call. See the field notes in CoasterBench's `evals/ci/README.md`.

## Gate

The test asserts protocol integrity, not coaster quality: every round must
complete with a submitted, game-accepted program. Whether a small model can
actually close a track circuit is a capability question and is reported, not
asserted.

## Running locally

```bash
# game binary: build CoasterBench, or extract from its game image
export COASTERBENCH_REPO=~/code/CoasterBench   # checkout with evals/driver.py
export COASTERBENCH_CLI=$COASTERBENCH_REPO/build/openrct2-cli

pytest -s -v tests/evals/coasterbench/test_coasterbench_protocol.py
```
