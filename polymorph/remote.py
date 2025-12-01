from __future__ import annotations

import os
import subprocess  # nosec B404
from pathlib import Path

from rich.console import Console

console = Console()


class RemoteConfig:
    def __init__(
        self, remote_user: str, remote_host: str, remote_path: str = "~/polymorph", ssh_key: str | None = None
    ):
        self.remote_user = remote_user
        self.remote_host = remote_host
        self.remote_path = remote_path
        self.ssh_key = ssh_key

    @property
    def remote_address(self) -> str:
        return f"{self.remote_user}@{self.remote_host}"

    @classmethod
    def from_env(cls) -> RemoteConfig | None:
        remote_user = os.getenv("POLYMORPH_REMOTE_USER")
        remote_host = os.getenv("POLYMORPH_REMOTE_HOST")

        if not remote_user or not remote_host:
            return None

        return cls(
            remote_user=remote_user,
            remote_host=remote_host,
            remote_path=os.getenv("POLYMORPH_REMOTE_PATH", "~/polymorph"),
            ssh_key=os.getenv("POLYMORPH_SSH_KEY"),
        )


def sync_code(config: RemoteConfig, local_path: Path) -> bool:
    exclude_patterns = [
        ".venv",
        "__pycache__",
        ".git",
        "data/",
        "*.pyc",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "*.egg-info",
        "dist/",
        "build/",
    ]

    rsync_cmd = ["rsync", "-az", "--delete"]

    for pattern in exclude_patterns:
        rsync_cmd.extend(["--exclude", pattern])

    if config.ssh_key:
        rsync_cmd.extend(["-e", f"ssh -i {config.ssh_key} -T"])

    rsync_cmd.extend(
        [
            f"{local_path}/",
            f"{config.remote_address}:{config.remote_path}/",
        ]
    )

    console.print(f"[dim]→ Syncing to {config.remote_host}...[/dim]", end=" ")
    result = subprocess.run(rsync_cmd, capture_output=True, check=False)  # nosec B603

    if result.returncode not in [0, 23, 24]:
        console.print("[red]✗[/red]")
        console.print(result.stderr.decode() if result.stderr else "")
        return False

    console.print("[green]✓[/green]")
    return True


def execute_remote(config: RemoteConfig, command_args: list[str]) -> int:
    polymorph_cmd = " ".join(command_args)

    commands = [
        f"cd {config.remote_path}",
        "source .venv/bin/activate",
        f"polymorph {polymorph_cmd}",
    ]

    remote_cmd = " && ".join(commands)

    ssh_cmd = ["ssh", "-t"]

    if config.ssh_key:
        ssh_cmd.extend(["-i", config.ssh_key])

    ssh_cmd.extend([config.remote_address, remote_cmd])

    console.print(f"[dim]→ Running on {config.remote_host}[/dim]\n")
    result = subprocess.run(ssh_cmd, check=False)  # nosec B603

    return result.returncode


def deploy_and_run(command_args: list[str]) -> int:
    config = RemoteConfig.from_env()

    if not config:
        console.print("[red]Error: Remote server not configured[/red]")
        console.print("\nSet these environment variables:")
        console.print("  export POLYMORPH_REMOTE_USER=your_username")
        console.print("  export POLYMORPH_REMOTE_HOST=your.server.com")
        console.print("  export POLYMORPH_REMOTE_PATH=~/polymorph  # optional")
        console.print("  export POLYMORPH_SSH_KEY=~/.ssh/id_rsa    # optional")
        return 1

    current_path = Path.cwd()
    while current_path != current_path.parent:
        if (current_path / "pyproject.toml").exists():
            break
        current_path = current_path.parent
    else:
        current_path = Path.cwd()

    if not sync_code(config, current_path):
        console.print("[red]Failed to sync code[/red]")
        return 1

    return execute_remote(config, command_args)
