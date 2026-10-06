# OCOP

Outcome-Conditioned Organizational Policy 的工程研究原型，用于管理、运行和分析多智能体组织实验。
PolicyLLM 根据任务、组织规则和目标成功率 `z`，生成四个 worker 的角色分配与通信 DAG。

主要流程：StrongLLM 构图 → 公共 executor 执行与标注 → PolicyLLM raw SFT → checkpoint 重载 → 多 `z` 评估与报告。
`z` 表示期望成功率，实际结果由执行测得；工程流程跑通与研究假设成立分别评估。

## 安装

使用 uv 管理环境，仓库默认 Python 3.12。在仓库根目录运行：

```bash
uv sync --locked
uv run --locked ocop --help
```

本地模型、全参数训练和评估需要 NVIDIA GPU 及额外依赖：

```bash
uv sync --locked --extra local --extra training
```

组织独立复测等统计分析还需 `--extra analysis`。依赖版本由 `uv.lock` 固定。

## 配置

默认运行配置为 [configs/prototype.json](configs/prototype.json)。开始实验前检查：

- `policy.model_path`：本地 Qwen3.5-2B 权重目录，需单独准备。
- `artifacts_dir`：产物目录，默认 `artifacts/`。
- 任务划分、候选数、执行重复数，以及请求并发和预算。

默认配置通过 DeepSeek 提议组织，通过 OpenRouter 调用 GPT-4o-mini 执行 worker 和 finalizer。
远端调用需要以下环境变量：

| 变量 | 用途 |
|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek 服务凭据 |
| `OPENAI_API_KEY` | OpenRouter 服务凭据 |

也可使用 dotenv 文件，通过 `--credentials PATH` 指定；默认读取本地
`my_docs/secrets/credentials.env`，环境变量优先。该文件不随仓库分发。

## 快速开始

离线校验示例组织，无需 API 凭据或模型权重：

```bash
uv run --locked ocop graph validate examples/graph/valid.json
```

配置凭据后检查远端服务并查看记录；`smoke` 会实际调用模型服务：

```bash
uv run --locked ocop smoke --config configs/prototype.json --run-id service-check
uv run --locked ocop runs show artifacts/runs/service-check
```

新实验使用新的 run ID；恢复时沿用该 run 的配置快照和已有记录。
本地完整采集、训练与评估流程见 `my_docs/usage.md`，该私有手册不随仓库分发。

## 主要命令

| 命令 | 用途 |
|---|---|
| `graph` | 查看构图契约、schema，校验并重放轨迹 |
| `smoke` | 验证真实模型服务 |
| `execute` | 执行一个组织并判分 |
| `collect` | 采集 GSM8K 组织、重复执行与成功率标签 |
| `prepare-sft` | 准备并审计 raw SFT 样本 |
| `verify-sft` | 验证参数更新、checkpoint 保存与重载 |
| `train` | 全参数 SFT，支持 checkpoint 恢复 |
| `evaluate` | 比较基础模型与训练后模型在不同 `z` 下的表现 |
| `report` | 从已有评估记录重建报告与指标 |
| `runs` | 查看、导出或恢复 run 记录 |
| `verify` | 采集、训练、评估及扩量验收；executor 真实请求检查 |
| `diagnose` | 模型检查、组织图统计与策略结果诊断 |
| `experiment` | 固定图基线、扩量监督与 SFT 闭环编排 |

通过 `uv run --locked ocop <命令> --help` 查看参数。
长时间采集、训练和评估使用 tmux 持久化。

## 目录

| 路径 | 内容 |
|---|---|
| `src/ocop/` | 项目实现与可复用逻辑 |
| `configs/` | 运行与实验配置 |
| `examples/` | 构图协议和执行任务示例 |
| `tests/` | 自动化测试 |
| `outputs/`、`artifacts/` | 运行记录、数据、checkpoint 和报告，不纳入 Git |
| `my_docs/` | 本地研究设计与交接资料，不纳入 Git |

本地研究定义和实现约定分别见 `my_docs/design/design.md` 与
`my_docs/design/Implement-design.md`。

`src/ocop/` 按 `graph`、`execution`、`collection`、`training`、`evaluation`、
`inference`、`runtime`、`diagnostics`、`experiments` 组织。实验逻辑位于所属 module；
所有用户入口统一为 `ocop`。测试按对应职责分组，共享测试数据和 fixtures 位于 `tests/support.py`。

已有模型可供新评估读取；历史训练产物和生产身份保留。旧 run 的原样复验或续跑使用冻结版本。

## 开发验证

```bash
uv run --locked --extra local --extra training --extra analysis pytest
```

若服务器预设的 CUDA 库路径与 uv 环境冲突，相关命令可加 `env -u LD_LIBRARY_PATH` 前缀；
环境检查与 GPU 要求见使用手册。
