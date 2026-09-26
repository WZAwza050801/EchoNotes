#!/usr/bin/env python3
"""环境自检 —— 跑管线之前先执行它。

用法:
    python scripts/check_env.py          # 全量检查（缺硬依赖时退出码 1）
    python scripts/check_env.py --ci     # 只校验 Python 版本与 pip 依赖（CI 用）
    python scripts/check_env.py -v       # 附带版本/路径细节

设计原则：环境问题不应该留给使用者去猜。缺什么、去哪装、装完怎么验证，一次说清。
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

# .env 与管线入口共用同一份加载逻辑（core.load_dotenv）；本脚本能独立于包运行，
# 这里直接把仓库根目录加入 path 后按模块名导入（core 只依赖标准库）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core  # noqa: E402

core.load_dotenv()

# (import 名, pip 名或 None)
PIP_REQUIRED = [('PIL', 'Pillow')]
# (import 名, 说明)
PIP_OPTIONAL = [('faster_whisper', '本地 ASR，可用 --transcript 复用现成转写跳过')]
# (命令, 是否必需, 安装提示)
SYS_TOOLS = [('ffmpeg', True, 'apt install ffmpeg / winget install Gyan.FFmpeg / brew install ffmpeg'), ('xelatex', True, 'apt install texlive-xetex texlive-lang-chinese / 安装 MiKTeX 或 TeX Live')]
# (环境变量, 是否必需, 用途)。run 主流程实际读取的角色是 TEXT 与 VISION；
# 单独设置 SILICONFLOW_API_KEY 不足以运行（见 docs/API_SETUP.md 的角色说明）。
ENV_KEYS = [
    ('ECHONOTES_TEXT_API_KEY', True, '转写整理/reduce（run 主流程必需）'),
    ('ECHONOTES_VISION_API_KEY', True, '窗口 map 与公式复查（run 主流程必需）'),
    ('ECHONOTES_SECRETS_FILE', False, '密码书路径；设置了它可替代上面两个变量'),
    ('SILICONFLOW_API_KEY', False, '仅当 ECHONOTES_VISION_PROVIDER=siliconflow 时作为视觉 Key；单独设置不足以运行'),
]

MIN_PYTHON = (3, 11)

OK, WARN, BAD = "[ OK ]", "[WARN]", "[FAIL]"


def run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return (r.stdout or r.stderr or "").strip().splitlines()[0] if (r.stdout or r.stderr) else ""
    except Exception:
        return ""


def check_python() -> tuple[bool, str]:
    v = sys.version_info
    ok = (v.major, v.minor) >= MIN_PYTHON
    return ok, f"Python {v.major}.{v.minor}.{v.micro} ({sys.executable})" + (
        "" if ok else f"  -> 需要 >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
    )


def check_pip(name: str, pip_name: str | None) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "")
        return True, f"{pip_name or name} {ver}".strip()
    except Exception:
        hint = f"  -> pip install {pip_name}" if pip_name else ""
        return False, f"缺少 {pip_name or name}{hint}"


def check_tool(cmd: str) -> tuple[bool, str]:
    path = shutil.which(cmd)
    if not path:
        return False, f"PATH 中找不到 {cmd}"
    return True, path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", action="store_true", help="只校验 Python 与 pip 依赖")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    print(f"\n=== {Path(__file__).resolve().parents[1].name} 环境自检 ===")

    hard_fail = False

    ok, info = check_python()
    print(f"{OK if ok else BAD} Python      : {info}")
    hard_fail |= not ok

    print("-- pip 依赖 -------------------------------")
    for name, pip_name in PIP_REQUIRED:
        ok, info = check_pip(name, pip_name)
        print(f"{OK if ok else BAD} {name:<12}: {info}")
        hard_fail |= not ok

    for name, note in PIP_OPTIONAL:
        ok, info = check_pip(name, name)
        print(f"{OK if ok else WARN} {name:<12}: {info if ok else '未安装（' + note + '）'}")

    if not args.ci:
        print("-- 系统工具 -------------------------------")
        for cmd, required, hint in SYS_TOOLS:
            ok, info = check_tool(cmd)
            if ok:
                ver = run([cmd, "--version"])
                print(f"{OK} {cmd:<12}: {ver or info}")
            else:
                print(f"{BAD if required else WARN} {cmd:<12}: 缺失 -> {hint}")
                hard_fail |= required

        print("-- API 密钥 -------------------------------")
        secrets_configured = bool(os.environ.get("ECHONOTES_SECRETS_FILE"))
        for key, required, why in ENV_KEYS:
            val = os.environ.get(key)
            # 角色变量（TEXT/VISION）在密码书已配置时视为满足
            satisfied = bool(val) or (secrets_configured and key.endswith("_API_KEY"))
            if satisfied:
                detail = f"已设置（{len(val)} 字符，不回显）" if val else "由 ECHONOTES_SECRETS_FILE 提供"
                print(f"{OK} {key:<24}: {detail}")
            else:
                print(f"{BAD if required else WARN} {key:<24}: 未设置（{why}）")
                hard_fail |= required

    print("-" * 48)
    if hard_fail:
        print("结论：存在必需项缺失，按上面 -> 提示补齐后再跑管线。")
        return 1
    print("结论：必需项齐备。可选未装项只影响对应功能，不影响主流程。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
