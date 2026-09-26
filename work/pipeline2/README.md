# 管线二使用手册：课程视频 → LaTeX 讲义

> 设计理念、证据纪律与架构说明见[主 README](../../README.md)。本手册只讲操作：
> 命令、参数、环境变量、缓存规则与排障。
>
> 模型层（core/models/writing）与独立仓库 [Ech_lecture](https://github.com/WZAwza050801/Ech_lecture)
> 同源，2026-09-27 已对齐其 `cf036f1`（重试分流 / reduce 重试 / EXTRA_BODY 护栏 /
> 节流时钟落盘）；后续模型层修复以两边同步提交为准。

## 运行环境

Python 3.11+；`ffmpeg`、`ffprobe`、`XeLaTeX` 在 PATH 中。
仓库根目录的 `requirements.txt` 是权威依赖清单（含安装说明与外部可执行文件的
各平台安装命令），推荐从根目录安装：

```powershell
python -m pip install -r requirements.txt          # 仓库根目录
python -m work.pipeline2.pipeline2 doctor   # 自检 ffmpeg/ffprobe/xelatex/PIL/faster_whisper
```

> 本目录下的 `requirements.txt` 为历史兼容保留，内容与根目录一致。

```powershell
python -m pip install -r work/pipeline2/requirements.txt
python -m work.pipeline2.pipeline2 doctor   # 自检 ffmpeg/ffprobe/xelatex/PIL/faster_whisper
```

密钥通过外部 JSON 文件提供（沿用管线一注册表格式，仓库不含任何密钥）：

```json
{"entries":[{"provider":"deepseek","label":"...","apiKey":"...","baseUrl":"...","models":["deepseek-chat"]}]}
```

## 命令

### run：课程视频 → 证据讲义

```powershell
python -m work.pipeline2.pipeline2 run 'https://www.bilibili.com/video/BV.../?p=2' `
  --secrets '外部密钥文件路径'
```

| 参数 | 说明 |
|---|---|
| `source` | BV 号、完整 `bilibili.com/video/BV...` 链接（`?p=N` 选分 P）或本地视频路径 |
| `--page N` | 选择分 P（与 URL `?p=N` 等价） |
| `--title` | 覆盖成品标题 |
| `--transcript 文件` | 已有时间戳转写 JSON，跳过下载与 ASR；格式 `{"segments":[{"start":0,"end":3.2,"text":"..."}]}` |
| `--interval 秒` | 均匀抽帧间隔，默认 30；密集板书可调 10 |
| `--max-frames N` | 帧预算，默认 240；长课按 `时长/interval` 估算并调大（超预算会显式报错，不静默降密度） |
| `--scene-threshold` | 场景检测阈值，默认 0.12 |
| `--window-seconds` | map 窗口时长，默认 600 |
| `--max-images` | 每窗口图片上限，默认 8 |
| `--prepare-only` | 只做下载+转写+抽帧，不调用云模型 |
| `--no-polish / --no-verify / --no-compile` | 独立关闭格式整理 / 公式复查 / PDF 编译 |
| `--keep-cache` | 成功后保留运行目录（默认成功即自动清理） |
| `--asr-python` / `--asr-model` | 复用已装 faster-whisper 的解释器与模型目录 |
| `--secrets` | 外部密钥 JSON，默认读 `ECHONOTES_SECRETS_FILE` |

### distill：把已有运行目录重新打包成成品夹

```powershell
python -m work.pipeline2.pipeline2 distill '运行目录或旧归档/lecture.json' --output-root '归档根'
```

成品夹（`BV号-P页-课程名/`）只含：`lecture.pdf`、`lecture.tex`、`lecture.json`、
`frames/`（去重后讲义引用的截图）与自动生成的 `README.md`。
编译中间产物不进成品夹；`run` 成功收尾会自动清理运行目录（`--keep-cache` 退出）。

### study：学习讲义（出版版 + 卡片版 + 概念地图）

```powershell
python -m work.pipeline2.study '成品夹或运行目录/lecture.json' `
  --output-root 'output/学习讲义' --secrets '外部密钥文件路径'
```

四道工序（初稿 → 出版级深化 → 数学审校 → 文风统一）全部带缓存；
课程概念地图页印在目录前。产出 `学习讲义.pdf`（出版编排）与
`学习讲义-卡片版.pdf`（彩色知识卡），内容同源。设计依据见 `LEARNING_DESIGN.md`。

## 环境变量

角色前缀：`TEXT`（转写清洗）、`VISION`（板书/公式）、`PLANNER`（学习规划）、
`WRITER`(学习写作)。每角色可用 `ECHONOTES_<ROLE>_PROVIDER / MODEL / BASE_URL /
API_KEY / KEY_LABEL / MAX_TOKENS / TEMPERATURE`。

| 变量 | 说明 |
|---|---|
| `ECHONOTES_SECRETS_FILE` | 外部密钥注册表路径 |
| `ECHONOTES_ASR_PYTHON` | 已装 faster-whisper 的解释器（主程序只需 Pillow） |
| `ECHONOTES_ASR_MODEL` | 本地模型目录（如 `models/faster-whisper-small`）或 `small` |
| `ECHONOTES_ASR_CPU_THREADS` / `ECHONOTES_ASR_BEAM_SIZE` | 默认 `2` / `3`；内存紧张设 `1` / `1` |
| `ECHONOTES_ASR_CHUNK_SECONDS` | ASR 分块秒数，默认 600；长课防内存溢出 |
| `ECHONOTES_MODEL_RETRIES` | 模型请求重试次数，默认 3 |
| `ECHONOTES_MODEL_BACKOFF` | 重试退避基数（秒），默认 10，按次数阶梯递增 |
| `ECHONOTES_MODEL_TIMEOUT` | 单次请求超时（秒），默认 180；密集板书窗口建议 600 |
| `ECHONOTES_MODEL_MIN_INTERVAL` | 同一角色两次请求的最小间隔（秒），默认 0；限流账号设 21。时钟落盘到运行目录（`cache/.request-clock.json`），重启续跑仍计时；N 个并行进程合计 RPM 约乘 N，请加倍或错峰 |
| `ECHONOTES_<ROLE>_EXTRA_BODY` | provider 私有参数逃生舱（JSON 对象）。不得覆盖保留键 `model/messages/stream/response_format/temperature/max_tokens`，否则实际请求与缓存身份脱节——启动时即报错并指明变量名 |
| `BILIBILI_COOKIE` | 完整 cookie 请求头；未设置时读取相邻管线一源码中的既有 cookie 字面量 |
| `DEEPSEEK_API_KEY` / `OPENROUTER_API_KEY` | 无注册表时的兜底 |

`.env` 放在 `work/pipeline2/.env`（包目录），启动时自动加载，shell 里已设置的同名变量优先。

代码默认值：文本/规划/写作者 `deepseek-v4-pro`（DeepSeek 直连），清洗 `deepseek-chat`，
视觉 `qwen/qwen3-vl-235b-a22b-instruct`（OpenRouter）；全部可用环境变量覆盖。
plan 类端点（百炼、Kimi Code 等）通过 `PROVIDER/BASE_URL/KEY_LABEL/MODEL` 组合接入，
密钥 label 需与注册表精确匹配。

## 缓存与断点

- 缓存键 = 输入、模型、prompt、参数与帧内容摘要；参数或证据变化自动重跑对应步骤。
- **缓存迁移规则**：早期版本不把 temperature/extra_body 计入缓存身份。曾设过非默认
  `ECHONOTES_<ROLE>_TEMPERATURE` / `_EXTRA_BODY` 的用户升级到计入这些字段的版本后，
  旧缓存会失配重打（一次性成本，属预期行为）；从未改过这两项的用户旧缓存继续命中。
- 同一视频有运行锁（`.running`）；异常退出自动清锁，进程被强杀后需确认旧进程
  已停再手动删除锁文件。
- ASR 每 50 段写 `asr-output.partial.json` 检查点（仅供诊断，成功后合并为正式缓存）。
- 模型 JSON 编码失败时最多一次纯编码修复；再次失败即停止，不静默接受损坏公式。
- 成功收尾：打包成品夹 → 校验 PDF 存在 → 删除运行目录（`--keep-cache` 保留）。

## 排障手册

| 症状 | 原因与处理 |
|---|---|
| B 站接口 412 | 极简请求头会被风控；保持完整浏览器头 + 完整 cookie 组，禁用系统代理 |
| `Uniform sampling needs N frames` | 帧数超预算；`--max-frames` 调大或 `--interval` 调稀 |
| ASR `Unable to allocate ... GiB` | 长音频整段 STFT 撑爆内存；升级到分块版 asr.py（本仓库已内置）或调小 `ECHONOTES_ASR_CHUNK_SECONDS` |
| `Model network request failed` | 看异常详情；沙箱/代理环境先清 `HTTP(S)_PROXY`；加大 `ECHONOTES_MODEL_RETRIES/BACKOFF/TIMEOUT` |
| `Misplaced alignment tab character &` | 模型输出丢矩阵环境反斜杠；检查缓存 JSON 中 `(?<![\\\w])(begin\|end){` 模式并补 `\\` |
| XeLaTeX 失败 | 看 `compile-lecture-*.txt`；未知数学命令会显示原文与待排版提示 |
| `Run is locked` | 确认旧进程已死后删除运行目录下 `.running` |
| 编译产物归档报 WinError 2 | 旧版归档清单与参数化编译产物名不一致（已修复于 ebc388a） |
| 讲义重编译失败 | `lecture.tex` 需要同目录 `frames/`；成品夹已自带 |

## 验证

```powershell
python -m unittest work.pipeline2.tests.test_core work.pipeline2.tests.test_models `
  work.pipeline2.tests.test_media work.pipeline2.tests.test_pipeline `
  work.pipeline2.tests.test_cloud_asr work.pipeline2.tests.test_study

# 真 ffmpeg + 真 XeLaTeX、固定假模型的端到端演示
python -m work.pipeline2.tests.test_pipeline 'output/pipeline2-integration'
```

## English quick reference

- `run <BV|URL|file> [--page N] [--title T] [--transcript F] [--interval S]
  [--max-frames N] [--prepare-only] [--keep-cache] [--secrets F]` — full pipeline:
  fetch, chunked local ASR, sampled+deduped frames, per-window multimodal map,
  reduce, formula re-verification, XeLaTeX compile, then package a clean
  per-course folder (`<BV>-P<page>-<title>/` with PDF/tex/lecture.json/frames/README)
  and auto-delete the run directory on success.
- `distill <path/to/lecture.json> [--output-root DIR]` — repackage an existing run
  or old archive into the clean per-course folder.
- `study <path/to/lecture.json> [--output-root DIR]` — study handout: four cached
  writing passes, concept map page, publication + card edition PDFs.
- `doctor` — check ffmpeg/ffprobe/xelatex/PIL/faster_whisper availability.
- Keys: external JSON registry via `--secrets` or `ECHONOTES_SECRETS_FILE`; per-role
  overrides `ECHONOTES_{TEXT|VISION|PLANNER|WRITER}_{PROVIDER,MODEL,BASE_URL,...}`;
  resilience via `ECHONOTES_MODEL_{RETRIES,BACKOFF,TIMEOUT}` and
  `ECHONOTES_ASR_CHUNK_SECONDS`.
- Evidence discipline: every block/formula carries `segment_ids`/`frame_ids`;
  fabricated IDs are rejected; uncertain formulas are re-checked against their
  source frames and never silently corrected. A successful compile does not imply
  mathematical correctness.
