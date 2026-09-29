"""Produces ONE real run, written to the SAME data/benchmark.sqlite the web
dashboard reads, with a live AcousticStreamProcessor attached and fed a
real audio fixture — so opening /runs/{run_id} in the browser shows
genuine acoustic data (FRUSTRATION/UNCERTAINTY/ENERGY lanes) produced by
the real pipeline, not something faked into the UI layer for a screenshot.

Usage: python -m benchmarks.acoustic_demo_run
Then: cd apps/web && npm run dev, and open the printed URL.
"""
from __future__ import annotations

import asyncio
import sys
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

from audio_io import load_fixture  # noqa: E402
from replay_runner import run_one  # noqa: E402
from scenarios import SCENARIOS_BY_ID  # noqa: E402


async def main() -> None:
    db_path = REPO_ROOT / "data" / "benchmark.sqlite"
    recorder = SqliteRecorder(db_path)
    scenario = SCENARIOS_BY_ID["long_monologue"]
    config = BackchannelConfig.load()
    acoustic_audio = load_fixture("increasing_frustration")

    acoustic_processor = AcousticStreamProcessor(
        config=AcousticConfig(), model=DeterministicProsodyModel(), recorder=recorder,
    )
    await acoustic_processor.start()

    row = await run_one(
        scenario, "backchannel", config, recorder, run_index=9999, seed=7,
        llm=make_llm(), tts=make_tts(),
        acoustic_processor=acoustic_processor, acoustic_audio=acoustic_audio,
    )
    await acoustic_processor.stop()
    recorder.close()

    print(f"run_id: {row['run_id']}")
    print(f"response_latency_ms: {row['response_latency_ms']:.1f}")
    print(f"Open: http://localhost:3000/runs/{row['run_id']}")


if __name__ == "__main__":
    asyncio.run(main())
