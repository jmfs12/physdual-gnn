"""Coleta de informações de diagnóstico do ambiente (SO, Python, Torch/CUDA, GPU, RAM, Disco, Colab)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from physdual.utils.paths import get_project_root, get_repo_root, is_colab


def _bytes_to_gb(num_bytes: int | float | None) -> float | None:
    """Converte bytes para gigabytes (GiB) com duas casas decimais."""
    if num_bytes is None:
        return None
    return round(float(num_bytes) / (1024**3), 2)


def _get_ram_gb() -> dict[str, float | None]:
    """Obtém RAM total e disponível em GiB (via psutil, /proc/meminfo ou ctypes no Windows)."""
    try:
        import psutil  # type: ignore[import-untyped]

        vm = psutil.virtual_memory()
        return {"total_gb": _bytes_to_gb(vm.total), "available_gb": _bytes_to_gb(vm.available)}
    except ImportError:
        pass

    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            data: dict[str, int] = {}
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val_kb = int(parts[1].strip().split()[0])
                    data[key] = val_kb * 1024
            return {
                "total_gb": _bytes_to_gb(data.get("MemTotal")),
                "available_gb": _bytes_to_gb(data.get("MemAvailable", data.get("MemFree"))),
            }
        except Exception:
            pass

    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return {
                "total_gb": _bytes_to_gb(stat.ullTotalPhys),
                "available_gb": _bytes_to_gb(stat.ullAvailPhys),
            }
        except Exception:
            pass

    return {"total_gb": None, "available_gb": None}


def _get_disk_usage(path: Path) -> dict[str, Any]:
    """Devolve espaço total, usado e livre (GiB) caso o caminho exista."""
    try:
        target = path if path.exists() else path.parent
        if not target.exists():
            return {"mounted": False, "path": str(path)}
        usage = shutil.disk_usage(target)
        return {
            "mounted": True,
            "path": str(path),
            "total_gb": _bytes_to_gb(usage.total),
            "used_gb": _bytes_to_gb(usage.used),
            "free_gb": _bytes_to_gb(usage.free),
        }
    except Exception as exc:
        return {"mounted": False, "path": str(path), "error": str(exc)}


def _run_nvidia_smi() -> str:
    """Executa `nvidia-smi` de forma segura e retorna a saída resumida."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
        return out.strip() or "Nenhuma GPU listada pelo nvidia-smi"
    except FileNotFoundError:
        return "nvidia-smi não encontrado no PATH"
    except Exception as exc:
        return f"nvidia-smi indisponível ({exc})"


def collect_env_info() -> dict[str, Any]:
    """Coleta informações completas de hardware, sistema operacional e bibliotecas."""
    torch_info: dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda_version": None,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
    }
    try:
        import torch

        torch_info["installed"] = True
        torch_info["version"] = torch.__version__
        torch_info["cuda_version"] = torch.version.cuda
        torch_info["cuda_available"] = bool(torch.cuda.is_available())
        if torch_info["cuda_available"]:
            count = torch.cuda.device_count()
            torch_info["device_count"] = count
            devices = []
            for idx in range(count):
                props = torch.cuda.get_device_properties(idx)
                devices.append(
                    {
                        "index": idx,
                        "name": props.name,
                        "vram_total_gb": _bytes_to_gb(props.total_memory),
                        "capability": f"{props.major}.{props.minor}",
                    }
                )
            torch_info["devices"] = devices
    except ImportError:
        pass

    colab_running = is_colab()
    local_disk_target = Path("/content") if Path("/content").exists() else get_repo_root()
    drive_mount = Path("/content/drive/MyDrive")
    drive_mounted = drive_mount.exists()

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "in_colab": colab_running,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "torch": torch_info,
        "nvidia_smi": _run_nvidia_smi(),
        "ram": _get_ram_gb(),
        "disk_local": _get_disk_usage(local_disk_target),
        "disk_drive": _get_disk_usage(drive_mount) if drive_mounted else {"mounted": False, "path": str(drive_mount)},
        "project_root": str(get_project_root()),
        "repo_root": str(get_repo_root()),
    }


def format_env_report(info: dict[str, Any] | None = None) -> str:
    """Formata o dicionário de diagnóstico do ambiente em texto legível."""
    data = info if info is not None else collect_env_info()
    lines = [
        "============================================================",
        "PhysDual-GNN — Diagnóstico do Ambiente",
        "============================================================",
        f"Data/Hora (UTC)    : {data['timestamp_utc']}",
        f"Rodando no Colab   : {data['in_colab']} ('google.colab' in sys.modules)",
        f"Sistema Operacional: {data['os']['platform']} ({data['os']['machine']})",
        f"Python             : {data['python']['version']} ({data['python']['executable']})",
        f"Raiz do Projeto    : {data['project_root']}",
        f"Raiz do Repositório: {data['repo_root']}",
        "------------------------------------------------------------",
    ]

    t = data["torch"]
    if t["installed"]:
        lines.append(f"PyTorch            : {t['version']} (CUDA compilado: {t['cuda_version']})")
        lines.append(f"CUDA Disponível    : {t['cuda_available']} (GPUs: {t['device_count']})")
        for dev in t["devices"]:
            lines.append(
                f"  - GPU [{dev['index']}]        : {dev['name']} | VRAM: {dev['vram_total_gb']} GiB (CC {dev['capability']})"
            )
    else:
        lines.append("PyTorch            : NÃO INSTALADO")

    lines.append(f"nvidia-smi         : {data['nvidia_smi']}")

    ram = data["ram"]
    lines.append(f"RAM do Sistema     : total={ram['total_gb']} GiB | disponível={ram['available_gb']} GiB")

    dl = data["disk_local"]
    if dl.get("mounted"):
        lines.append(
            f"Disco Local        : {dl['path']} (livre={dl['free_gb']} GiB / total={dl['total_gb']} GiB)"
        )
    else:
        lines.append(f"Disco Local        : indisponível ({dl.get('path')})")

    dd = data["disk_drive"]
    if dd.get("mounted"):
        lines.append(
            f"Google Drive       : MONTADO em {dd['path']} (livre={dd['free_gb']} GiB / total={dd['total_gb']} GiB)"
        )
    else:
        lines.append(f"Google Drive       : NÃO MONTADO ({dd.get('path')})")

    lines.append("============================================================")
    return "\n".join(lines)


def write_env_snapshot(output_path: str | Path | None = None) -> Path:
    """Grava o relatório de ambiente e a saída de `pip freeze` em `docs/colab_env_snapshot.txt`."""
    target = (
        Path(output_path)
        if output_path is not None
        else (get_repo_root() / "docs" / "colab_env_snapshot.txt")
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    report = format_env_report()
    try:
        pip_freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        ).strip()
    except Exception as exc:
        pip_freeze = f"# Erro ao executar pip freeze: {exc}"

    content = f"{report}\n\n# --- pip freeze ---\n{pip_freeze}\n"
    target.write_text(content, encoding="utf-8")
    return target
