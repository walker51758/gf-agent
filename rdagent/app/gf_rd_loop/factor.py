"""
GF (GuangFa Securities) factor workflow with session control.

The loop logic is identical to the Qlib factor loop; only the configuration (and
therefore the scenario / data / LLM backend) is GF-specific.
"""

import asyncio
from pathlib import Path
from typing import Any, Optional

import fire
import typer
from typing_extensions import Annotated

from rdagent.app.gf_rd_loop.conf import GF_FACTOR_PROP_SETTING, configure_gf_llm_backend
from rdagent.app.qlib_rd_loop.factor import FactorRDLoop
from rdagent.core.exception import FactorEmptyError
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.gf_quant.data.loader import ensure_gf_data


class GFFactorRDLoop(FactorRDLoop):
    """GF factor R&D loop (reuses the whole Qlib factor loop)."""

    skip_loop_error = (FactorEmptyError,)


def main(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: str | None = None,
    checkout: Annotated[bool, typer.Option("--checkout/--no-checkout", "-c/-C")] = True,
    checkout_path: Optional[str] = None,
):
    """
    Auto R&D Evolving loop for GF Securities factors.

    You can continue running session by

    .. code-block:: python

        dotenv run -- python rdagent/app/gf_rd_loop/factor.py $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional paramter

    """
    # 1) Wire the GF LLM backend (DeepSeek via LiteLLM)
    configure_gf_llm_backend()

    # 2) Fail fast with a descriptive message when the GF factor data is missing
    ensure_gf_data(raise_on_missing=True)

    if not checkout_path is None:
        checkout = Path(checkout_path)

    if path is None:
        loop = GFFactorRDLoop(GF_FACTOR_PROP_SETTING)
    else:
        loop = GFFactorRDLoop.load(path, checkout=checkout)
    asyncio.run(loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
