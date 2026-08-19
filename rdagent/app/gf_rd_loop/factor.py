"""Remote-first GF (GuangFa Securities) RD-Agent factor loop."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

import typer
from dotenv import load_dotenv
from typing_extensions import Annotated

# Load the repository .env before importing module-level Pydantic settings.
load_dotenv()

from rdagent.app.gf_rd_loop.conf import GF_FACTOR_PROP_SETTING, configure_gf_llm_backend
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import FactorEmptyError
from rdagent.log import rdagent_logger as logger


class GFFactorRDLoop(RDLoop):
    """RD-Agent loop whose development stage executes the remote GF DSL."""

    skip_loop_error = (FactorEmptyError,)

    def running(self, prev_out: dict[str, Any]):
        exp = self.runner.develop(prev_out["coding"])
        if exp is None:
            raise FactorEmptyError("Remote GF factor evaluation returned no experiment")
        logger.log_object(exp, tag="runner result")
        return exp


def main(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: str | None = None,
    checkout: Annotated[bool, typer.Option("--checkout/--no-checkout", "-c/-C")] = True,
    checkout_path: Optional[str] = None,
):
    """Run the iterative GF factor research loop with remote evaluation."""
    configure_gf_llm_backend()

    if checkout_path is not None:
        checkout = Path(checkout_path)
    if path is None:
        loop = GFFactorRDLoop(GF_FACTOR_PROP_SETTING)
    else:
        loop = GFFactorRDLoop.load(path, checkout=checkout)
    asyncio.run(loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    typer.run(main)
