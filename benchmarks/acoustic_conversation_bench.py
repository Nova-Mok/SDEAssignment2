"""Phase 8/14 — Preserve conversational performance: acoustic OFF vs ON.

Uses the SAME trusted LLM/TTS-timing harness as replay_runner.py (real AWS
Bedrock + Polly calls, scripted STT timing, no live LiveKit room — see that
module's docstring for why) to answer the assignment's central question:

    Does continuously analyzing the user's voice make the normal agent slower?

"OFF" = exactly Assignment 1's `run_one()` — no acoustic processor
attached at all. "ON" = the SAME scenario, SAME backchannel config, with a
real audio fixture fed to a live AcousticStreamProcessor for the full
duration of the turn, exactly as worker.py wires it in production. Every
other measured quantity (response_latency_ms, llm_ttft_ms,
tts_first_audio_ms) comes from real AWS calls, identically in both
conditions — this script changes nothing else between the two runs.

Writes its own sqlite/JSON (not data/benchmark.sqlite — that file's schema
is shared with the web dashboard's baseline/backchannel comparison and
this is a different, additional comparison axis) so it can be re-run
without disturbing the existing benchmark history.

Usage: python -m benchmarks.acoustic_conversation_bench [--n 10] [--scenario long_monologue]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT / "apps" / "agent", REPO_ROOT / "benchmarks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from acoustic.config import AcousticConfig  # noqa: E402
from acoustic.mock_model import DeterministicProsodyModel  # noqa: E402
from acoustic.pipeline import AcousticStreamProcessor  # noqa: E402
from aws_providers import make_llm, make_tts  # noqa: E402
from backchannel.config import BackchannelConfig  # noqa: E402
from backchannel.instrumentation import SqliteRecorder  # noqa: E402

from aggregate import summarize  # noqa: E402
from audio_io import load_fixture  # noqa: E402
from replay_runner import run_one  # noqa: E402
from scenarios import SCENARIOS_BY_ID  # noqa: E402

METRICS = ("response_latency_ms", "llm_ttft_ms", "tts_first_audio_ms")


async def main(n: int, scenario_id: str) -> None:
    scenario = SCENARIOS_BY_ID[scenario_id]
    db_path = REPO_ROOT / "data" / "acoustic_conversation_bench.sqlite"
    db_path.unlink(missing_ok=True)
    recorder = SqliteRecorder(db_path)

    llm = make_llm()
    tts = make_tts()
    config = BackchannelConfig.load()
    acoustic_audio = load_fixture("increasing_frustration")

    results: dict[str, list[dict]] = {"off": [], "on": []}
    print(f"Running {n} x OFF and {n} x ON for scenario={scenario_id!r} (alternating)...")
    for i in range(n):
        for label in ("off", "on"):
            acoustic_processor = None
            if label == "on":
                acoustic_processor = AcousticStreamProcessor(
                    config=AcousticConfig(), model=DeterministicProsodyModel(),
                )
                await acoustic_processor.start()

            row = await run_one(
                scenario, "backchannel", config, recorder, run_index=i, seed=42 + i,
                llm=llm, tts=tts,
                acoustic_processor=acoustic_processor,
                acoustic_audio=acoustic_audio if label == "on" else None,
            )
            if acoustic_processor is not None:
                await acoustic_processor.stop()
            results[label].append(row)
            print(f"  [{label}] run {i}: response_latency_ms={row['response_latency_ms']:.1f}")

    report = {"scenario": scenario_id, "n": n, "metrics": {}}
    print(f"\n{'metric':<22} {'OFF p50':>10} {'ON p50':>10} {'delta p50':>10} {'OFF p95':>10} {'ON p95':>10} {'delta p95':>10}")
    for metric in METRICS:
        off_summary = summarize([r[metric] for r in results["off"]])
        on_summary = summarize([r[metric] for r in results["on"]])
        report["metrics"][metric] = {"off": off_summary, "on": on_summary}
        d50 = on_summary["p50"] - off_summary["p50"]
        d95 = on_summary["p95"] - off_summary["p95"]
        print(
            f"{metric:<22} {off_summary['p50']:>10.1f} {on_summary['p50']:>10.1f} {d50:>+10.1f} "
            f"{off_summary['p95']:>10.1f} {on_summary['p95']:>10.1f} {d95:>+10.1f}"
        )

    out_path = REPO_ROOT / "data" / "acoustic_conversation_bench.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")
    recorder.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--scenario", type=str, default="long_monologue")
    args = parser.parse_args()
    asyncio.run(main(args.n, args.scenario))
