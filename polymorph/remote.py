from __future__ import annotations

import subprocess  # nosec B404
import tempfile
import tomllib
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
    def from_config(cls) -> RemoteConfig | None:
        from polymorph.config import settings

        if not settings.remote.user or not settings.remote.host:
            return None

        return cls(
            remote_user=settings.remote.user,
            remote_host=settings.remote.host,
            remote_path=settings.remote.path or "/home/user/polymorph",
            ssh_key=settings.remote.ssh_key or "",
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
        "polymorph.toml",
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

    local_toml = local_path / "polymorph.toml"
    if local_toml.exists():
        console.print("[dim]→ Syncing config (remote→default)...[/dim]", end=" ")
        try:
            with open(local_toml, "rb") as f:
                config_data = tomllib.load(f)

            remote_config_data = config_data.get("remote", {})

            with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as tmp:
                tmp.write("[default]\n")
                for key, value in remote_config_data.items():
                    if isinstance(value, str):
                        tmp.write(f'{key} = "{value}"\n')
                    else:
                        tmp.write(f"{key} = {value}\n")
                tmp_path = tmp.name

            scp_cmd = ["scp"]
            if config.ssh_key:
                scp_cmd.extend(["-i", config.ssh_key])
            scp_cmd.extend([tmp_path, f"{config.remote_address}:{config.remote_path}/polymorph.toml"])

            result = subprocess.run(scp_cmd, capture_output=True, check=False)  # nosec B603
            Path(tmp_path).unlink()

            if result.returncode != 0:
                console.print("[red]✗[/red]")
                return False

            console.print("[green]✓[/green]")
        except Exception as e:
            console.print(f"[red]✗[/red] ({e})")
            return False

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
    config = RemoteConfig.from_config()

    if not config:
        console.print("[red]Error: Remote server not configured[/red]")
        console.print("\nConfigure in polymorph.toml [remote] section:")
        console.print('  user = "your_username"')
        console.print('  host = "your.server.com"')
        console.print('  path = "/home/your_username/polymorph"')
        console.print('  ssh_key = "/path/to/ssh/key"')
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
