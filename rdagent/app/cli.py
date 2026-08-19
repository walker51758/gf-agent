"""Command-line entry point for the GF factor agent."""

import typer

from rdagent.app.gf_rd_loop.factor import main as run_factor_loop

app = typer.Typer(help="GF A-share factor research agent")
app.command("factor")(run_factor_loop)


if __name__ == "__main__":
    app()
