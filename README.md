# OCOP

Outcome-Conditioned Organizational Policy 的工程 research prototype。
PolicyLLM 根据任务、组织规则和目标成功率生成四个 worker 的角色分配与通信 DAG。

## 当前功能

已实现版本化组织协议、固定角色 prompts、严格重放、SQLite/文件记录、请求恢复、真实服务 smoke，
以及公共 DAG executor、独立 finalizer、严格数字判分、GSM8K 采集与标签聚合、raw SFT 全参数训练和多 z 对照。
统一 CLI 的 `graph`、`smoke`、`execute`、`collect`、`runs`、`prepare-sft`、`verify-sft`、`train`、`evaluate`、`report` 命令可运行。
`prepare` 为预留入口，当前调用返回退出码 2。

## 环境与运行

使用 uv 与 Python 3.12。在仓库根目录运行：

```bash
uv sync --locked
uv run --locked ocop --help
uv run --locked ocop graph contract
uv run --locked ocop graph schema
uv run --locked ocop graph validate examples/graph/valid.json
uv run --locked ocop graph validate examples/graph/invalid.json
env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training pytest
```

`graph validate` 输出 JSON：原始 content/reasoning、契约版本及 hash、初始图、
每个有效动作后的图、最终图或首个错误。可使用 `--reasoning-file PATH` 传入独立原生 reasoning。
退出码：0 为完整合法轨迹；1 为解析或构图错误；2 为命令或文件读取错误。
这些命令不调用模型服务。

## Raw SFT 数据与训练链路验证

安装与检查使用 `--extra local --extra training`。本服务器运行 PyTorch 时需清除外部
`LD_LIBRARY_PATH`，以使用 uv 环境内匹配的 CUDA 库。

```bash
env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training ocop prepare-sft --source artifacts/runs/gsm8k-20260920 --output artifacts/sft/gsm8k-20260920-raw
env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training ocop verify-sft --data artifacts/sft/gsm8k-20260920-raw --output artifacts/sft/gsm8k-20260920-verification --device cuda:0
```

`prepare-sft` 只读核验采集库，将 train split 中合法且完整标注的 raw 轨迹写入 `samples.json`，
保留零成功率与重复图候选。样本关联原始记录 hash、候选、轨迹、图、标签和完整执行 ID；
参考答案和 eval 轨迹不进入训练输入。`manifest.json` 保存模型权重、tokenizer、模板、数据和配置指纹。
`length-report.json` 保存每条完整序列长度和超限名单。超限时退出码为 2，产物保留，训练入口拒绝加载；
输出目录必须是新目录，避免覆盖既有产物。

输入为组织规则和包含 question、z 的 JSON；z 使用实测 mean_outcome。
Qwen thinking 模板提供助手前缀，监督 reasoning、closing thinking delimiter、原始 content 和消息结束 token。
原始文本完整归档；送入模板时遵循模板自身的首尾空白规范化。每条样本保存 input_ids、labels、loss_groups
及边界；输入前缀、padding、消息结束后的分隔换行不计 loss。逐 token 审计在准备和启动验证时执行。
reasoning loss 包含思考结束分隔符；content loss 包含段内说明、动作和助手结束标记。

`verify-sft` 从 train split 确定性选择八个候选，包含最长样本及存在的零标签样本。
使用一张至少有 23 GiB 空闲显存的 GPU，microbatch=1、累积四次，完成两个全参数 AdamW 更新。
bf16 参数及 Adam moments、fp32 交叉熵；全模型参数均可训练，视觉分支没有文本梯度时如实记录。
按每个有效 batch 的监督 token 总数归一化；使用 PyTorch checkpoint 分块计算输出头交叉熵，
不改变目标序列或 loss 权重。验证设置集中在 `config/prototype.json` 的 `diagnostics.sft_verification`。

`checkpoint-step-2/` 保存模型、tokenizer、模板、optimizer/scheduler/RNG 和完整文件校验信息。
独立进程重载后检查参数 hash 与固定前缀 logits 完全一致，再恢复一次优化器更新并完成最多 256 token
的训练题生成检查。生成合法性、提前结束或达到长度上限都写入报告，生成合法性不作为工程通过条件。
`verification.json` 的 `passed=true` 表示两个更新及独立重载／恢复检查均通过；
`reload-report.json` 保存子进程证据，`failure.json` 或 `reload-failure.json` 保存失败诊断。
失败进程返回非零，不自动调整精度、数据预算或 GPU 配置。

TensorBoard 事件位于输出目录的 `tensorboard/` 和 `tensorboard-reload/`，记录 loss、分项 token 数、
梯度、优化器步数、显存、耗时和生成合法性：

```bash
uv run --locked --extra local --extra training tensorboard --logdir artifacts/sft/gsm8k-20260920-verification --host 127.0.0.1
```

长验证须通过 tmux 启动；先检查 GPU 空闲情况，保存进程日志。验证 checkpoint 与后续完整训练独立。

## 全参数 SFT

`ocop train` 使用完整 train split，从配置中的基础模型开始训练。当前为 72 条样本、3 epochs、
54 次优化器更新；每个 epoch 按候选 ID 排序后使用 `seed + epoch` 打乱，每条样本每轮出现一次。
输入数据与模型文件、模板、token mask 在启动时复核。正式运行参数集中在
`config/prototype.json` 的 `training_runtime`：microbatch=1、累积4次、AdamW、constant scheduler、
gradient clipping=1.0、每6步保存、保留全部 checkpoint。精度、loss 归一化与验证链路一致。
末尾不足4条的 batch 按其实际监督 token 总数归一化，不丢弃样本。

新训练输出目录必须不存在；使用一张至少有23 GiB空闲显存的 GPU。启动示例：

```bash
tmux -S /data1/zhilingyu/ocop/artifacts/sft-training.sock new-session -d -s ocop-sft -c /data1/zhilingyu/ocop 'env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training ocop train --data artifacts/sft/gsm8k-20260920-raw --output artifacts/sft/gsm8k-20260920-full --device cuda:0 > artifacts/gsm8k-20260920-sft-training.log 2>&1'
```

输出包括 `config.json`、`run-manifest.json`、`status.json`、`attempts/`、`tensorboard/` 和
`checkpoint-step-N/`。run manifest 保存数据、配置、代码与依赖指纹，以及完整训练 batch 顺序。
每次运行有独立 attempt，保留逐步指标及失败记录；文件锁防止同一输出目录被并发写入。

checkpoint 保存模型、tokenizer/template、optimizer/scheduler、随机数状态、下一 batch 位置及
有效更新历史，所有文件都有校验和。保存先写临时目录，校验完成后原子发布；未发布目录不会用于恢复。
`latest-checkpoint.json` 指向最近一次保存；恢复还会检查已发布目录，处理发布后、指针更新前的中断。
保存间隔内的未保存更新在恢复时重算。

恢复时把 `--resume-checkpoint` 指向**该 run 最新已发布 checkpoint**，例如：

```bash
tmux -S /data1/zhilingyu/ocop/artifacts/sft-training.sock new-session -d -s ocop-sft -c /data1/zhilingyu/ocop 'env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training ocop train --data artifacts/sft/gsm8k-20260920-raw --output artifacts/sft/gsm8k-20260920-full --device cuda:0 --resume-checkpoint artifacts/sft/gsm8k-20260920-full/checkpoint-step-6 >> artifacts/gsm8k-20260920-sft-training.log 2>&1'
```

恢复自动读取 run 内配置快照；显式传入的 `--config` 必须与快照一致。数据、代码、依赖、配置或
文件校验不匹配时拒绝恢复。已到最终步的 checkpoint 不再增加训练更新，可以重新执行最终验收。
如在首次 checkpoint 保存前中断，应使用新输出目录重新启动。

TensorBoard 使用同一条 global step 轴；恢复通过 `purge_step` 隐藏失效更新，原始 attempt 文件仍保留。
记录总体及 reasoning/content loss、分项 token 数、梯度、参数变化、学习率、显存、耗时、epoch 进度，
以及每轮按 token 加权的汇总 loss。训练结束自动释放模型，再由独立进程重载最终 checkpoint，
检查参数 hash、固定前缀 logits 和256 token训练题生成。生成的合法性、EOS及长度进入 TensorBoard。
TensorBoard 服务示例（端口占用时先检查已有服务）：

```bash
tmux -S /data1/zhilingyu/ocop/artifacts/sft-training.sock new-session -d -s ocop-tensorboard -c /data1/zhilingyu/ocop 'uv run --locked --extra local --extra training tensorboard --logdir artifacts/sft/gsm8k-20260920-full/tensorboard --host 127.0.0.1 --port 6006 > artifacts/gsm8k-20260920-tensorboard.log 2>&1'
```

通过 SSH 将服务器6006端口转发到本机后浏览。`training-report.json` 的 `passed=true` 表示
训练步数及样本覆盖、更新证据、全部 checkpoint 校验、独立重载与 TensorBoard 事件完整性均通过。
`reload-report.json` 保存重载和生成证据；`status.json` 区分 starting、training、verifying、completed、failed。
最后一个 checkpoint 用于后续训练前后与多 z 对照；训练题短生成仅作工程观察。

完成后可再次只读核验数据与配置指纹、样本覆盖、全部checkpoint文件、最终优化器状态、重载报告及
TensorBoard事件。验收输出写在训练目录之外，保留原训练报告：

```bash
env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training python scripts/verify_training.py artifacts/sft/gsm8k-20260920-full --output artifacts/gsm8k-20260920-sft-final-verification.json
```

## 扩量数据的 Pilot SFT

`config/gsm8k-pilot-sft.json` 使用已冻结的扩量 task manifest：200 道训练题和全部 50 道 holdout。
训练采用完整 raw 轨迹、原始候选的 R=5 标签与自然分布，从基础 Qwen3.5-2B 开始全参数 SFT。
1,121 条完整样本的长度为 1,497–6,338 tokens，均在 8192 上限内；3 epochs 共 843 次更新，
学习率 1e-5、有效 batch size 4，每100步及最后一步保存，共保留9个 checkpoint。

```bash
env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training ocop prepare-sft --config config/gsm8k-pilot-sft.json --source artifacts/runs/gsm8k-20260921-expansion-collection --output artifacts/sft/gsm8k-20261001-pilot-raw
env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training python scripts/run_pilot_sft.py --preflight-only
```

预检逐条复核 token mask、数据和模型指纹、任务隔离及评估矩阵，并在
`artifacts/experiments/gsm8k-20261001-pilot/` 冻结配置和 `preflight.json`。
完整流程复用现有验证、训练、评估与只读验收入口：先验证真实更新和独立重载，再训练、验收，
最后比较基础模型与最终 checkpoint 的 z∈{0,0.5,1}。
50题共300次构图，每个合法图五次完整执行；另外两次训练题smoke单列。
`evaluation.task_split="holdout"` 保留原始split及来源关联。组织生成、泛化和z响应分别报告。

```bash
tmux -S /data1/zhilingyu/ocop/artifacts/sft-training.sock new-session -d -s ocop-pilot -c /data1/zhilingyu/ocop 'env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training python scripts/run_pilot_sft.py > artifacts/gsm8k-20261001-pilot-pipeline.log 2>&1'
```

每个 GPU 阶段最多等待24小时，可用 `--gpu-wait-hours` 调整。
验证和训练需要23 GiB空闲显存，评估需要12 GiB；扫描最多6张GPU并选择空闲显存最多的可用卡。
`pipeline-status.json` 显示等待、当前阶段、失败或完成状态，阶段日志及最终验收报告保存在实验目录。
执行阶段的请求和repeat预算沿用配置；进程失败后按对应CLI入口恢复，历史记录和累计预算保留。

## 基础模型与最终 checkpoint 的多 z 对照

`ocop evaluate` 复用已完成采集run的task manifest和已验收训练run的最终checkpoint。
首次对照为6个eval任务 × 两个模型 × z∈{0,0.5,1}，共36次生成；再选按candidate ID排序的首条
训练样本，使用其原始z，两模型各做一次完整长度smoke，共38次。训练题smoke单列，六题eval属于开发评估。

Transformers推理使用单张空闲GPU、bf16、SDPA、单样本生成；启动要求至少12 GiB空闲显存。
先生成基础模型的19个候选，再加载最终checkpoint生成19个，释放GPU后执行合法图。
原生thinking、temperature=0.7、top_p=0.8、top_k=0、repetition_penalty=1、num_beams=1、最多8192个新token，
两模型的完整生成配置一致。每个任务／候选槽位由seed=42派生稳定种子，跨模型和z共用，生成前重设RNG。
输入只含规则、问题和指定z；参考答案仅用于执行评分。

启动前核验采集记录、数据与训练来源、最终checkpoint校验和、tokenizer/template、模型和executor指纹。
运行快照包含完整候选清单、配置、模型、依赖与源码指纹。启动示例：

```bash
tmux -S /data1/zhilingyu/ocop/artifacts/policy-evaluation.sock new-session -d -s ocop-eval -c /data1/zhilingyu/ocop 'env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training ocop evaluate --source artifacts/runs/gsm8k-20260920 --training-run artifacts/sft/gsm8k-20260920-full --run-id gsm8k-20260920-eval --device cuda:0 > artifacts/gsm8k-20260920-eval.log 2>&1'
```

`--phase generate`仅生成，不加载远端凭据；`--phase execute`要求所有候选已生成，消费现有图执行；
默认`--phase all`完成两者。合法图各取得5次完整执行，每图最多8次repeat；两个候选并行，请求并发8。
答错和答案格式错误计完整执行中的任务失败。38个图全部合法且顺利完成时，共190次图执行、950次远端调用。
重复图独立执行；生成数和repeat预算不会自动扩展。

SQLite和内容寻址文件位于`artifacts/runs/gsm8k-20260920-eval/`。先保存原始token IDs和文本，再严格解析；
只有正常EOS结束、thinking结束标记存在且完整动作轨迹合法的输出才能执行。截断和非法输出均保留。
`generation_attempt`记录本地生成尝试，`generation`保存完整原始输出，`trajectory`保存解析结果，
`evaluation_repeat`与`label`关联执行和评分；`evaluation-manifest.json`保存来源及候选配置。

恢复使用相同命令和run ID，省略`--config`时优先读取默认artifacts目录下该run的配置快照；自定义artifacts目录
应显式传入run内的`config.json`。恢复跳过已保存的生成和执行，原始输出已保存但未解析时只补解析。
生成中断沿用原候选和种子，保留旧尝试；CUDA/OOM等错误停止运行，记录`failure-*.json`。
在途预算共享等待和请求重试沿用现有实现；额度充值后的显式恢复使用`--resume-after-topup`，
旧在途预算fatal记录的恢复使用`--resume-inflight-budget`，保持历史结果和累计repeat预算。
修改数据、模型、代码或配置指纹需新run。

若因连续连接失败耗尽请求重试而停止，可在 tmux 中执行连接故障恢复脚本：

```bash
env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training python scripts/resume_evaluation.py --run artifacts/runs/gsm8k-20260920-eval
```

脚本核验冻结的数据／模型／源码、生成记录、触发熔断的请求证据及当前服务目录连通性，
失败序列允许 `RemoteProtocolError` 和 `ConnectError`，每个耗尽请求最后一次须为 `ConnectError`。
记录 `connect_error_recovery.v1` 恢复事件后解除熔断，继续执行剩余候选。
历史请求、生成、repeat及已终结标签保留，累计预算继续生效；已耗尽8次repeat的候选保持incomplete。
恢复脚本位于`scripts/`，其校验和保存在恢复事件中。仍不可连接或错误类型不匹配时停止。
若解除熔断后进程意外中断，使用原`ocop evaluate --phase execute`命令继续。

每次候选落盘会更新`evaluation-report.json`、`evaluation-report.md`、`records.jsonl`与TensorBoard。
报告分别列出训练题和eval、每个模型／z的生成计划、完成数、正常结束率、JSON解析率、合法图率、截断率、
可执行图、完整／不完整标签、成功率、正确执行覆盖、token长度、资源与usage，以及角色和边集合对照。
生成率以已完成生成为分母，正确执行覆盖以该组全部计划候选为分母；成功率仅使用获得5次完整执行的图。
没有可执行图的组保留“无可执行图”和空成功率；incomplete标签的成功率为空。

`finished`表示所有候选进入终态且run未halt；`integrity_passed`表示持久化记录能通过重放、重新判分和标签重算；
`policy_executor_evidence`表示至少一个重载模型生成的图获得可判分执行；`engineering_passed`要求三项同时满足。
生成全部非法也可以完成预算，报告如实记录工程实证状态。无需模型性能提升作为工程验收条件。

暂停或完成后，可从数据库重建报告并核验TensorBoard事件（需要获得run写锁，不修改数据库）：

```bash
env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training ocop report --run artifacts/runs/gsm8k-20260920-eval
```

完成后可运行只读最终验收，核对冻结输入、原始blob、评分与标签、TensorBoard及JSONL导出。
`--baseline`可指定恢复前报告，额外核验原始生成、既有repeat与终态标签保持一致；输出包含训练z覆盖统计。

```bash
env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training python scripts/verify_evaluation.py artifacts/runs/gsm8k-20260920-eval --baseline artifacts/gsm8k-20260920-eval-before-recovery.json --output artifacts/gsm8k-20260920-eval-final-verification.json
```

TensorBoard事件记录候选的固定ordinal和分组累计快照；恢复重建已有指标，不重复累计候选。
评估指标服务可独立监听本机6007端口，端口占用时先核查：

```bash
tmux -S /data1/zhilingyu/ocop/artifacts/policy-evaluation.sock new-session -d -s ocop-eval-tensorboard -c /data1/zhilingyu/ocop 'uv run --locked --extra local --extra training tensorboard --logdir artifacts/runs/gsm8k-20260920-eval/tensorboard --host 127.0.0.1 --port 6007 > artifacts/gsm8k-20260920-eval-tensorboard.log 2>&1'
```

## 固定图基线

固定图配置在`config/fixed-graph-baseline.json`：Decomposer、Solver、Checker、Reviser按worker顺序组成三边链。
入口读取已完成PolicyLLM评估的eval任务和executor配置，创建独立run，保存来源报告、图hash与脚本指纹。
每题5次完整执行、最多8次repeat；两题并发，原请求限制继续生效。当前六题正常共150次远端调用。

```bash
env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training python scripts/fixed_graph_baseline.py --source artifacts/runs/gsm8k-20260920-eval --run-id gsm8k-20260921-fixed-chain --prepare-only
env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training python scripts/fixed_graph_baseline.py --source artifacts/runs/gsm8k-20260920-eval --run-id gsm8k-20260921-fixed-chain
```

长任务在tmux中运行。以相同source、config和run ID恢复会复用已保存执行；完整候选不重复调用。
`--resume-after-topup`支持保存的额度错误恢复，保留历史请求和累计repeat预算。
生成`baseline-report.json`／`.md`、`records.jsonl`和`tensorboard/`，每个候选结束时更新。
报告按模型／z列出双方均有完整标签的共同任务、各自成功率及差值、同图对数，并记录全部计划候选的正确覆盖。
训练题smoke不进入基线，无可执行图的PolicyLLM组保留未测得结果，incomplete不进入完整标签成功率分母。

结束后重建报告、重新判分与聚合标签并检查TensorBoard（不新增执行）：

```bash
env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training python scripts/fixed_graph_baseline.py --report artifacts/runs/gsm8k-20260921-fixed-chain
env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training python scripts/verify_evaluation.py artifacts/runs/gsm8k-20260921-fixed-chain --output artifacts/gsm8k-20260921-fixed-chain-final-verification.json
```

最终验收入口自动识别固定图基线，额外复核冻结来源、所有blob、TensorBoard、JSONL与数据库一致性，
以及基线和原策略评估的请求独立性。验收保持数据库及已有报告不变。

基线的TensorBoard目录为`artifacts/runs/gsm8k-20260921-fixed-chain/tensorboard`，可在本机6008端口查看。

## 构图协议 v1

契约位于 `src/ocop/graph/contracts/v1.json`，版本为 `ocop.graph.v1`。
内含四个 worker ID、四种固定角色 prompt 和设计指令。`graph contract` 同时输出结构 schema
和覆盖契约内容及 schema 的 SHA-256。修改角色 prompt 或规则时需发布新契约版本。

Content 必须为单个 JSON 对象，字段恰为 `version` 和 `steps`。
每个 step 恰有非空白字符串 `explanation` 和一个 `action` 对象：

| 动作 | 字段 |
|---|---|
| `ASSIGN_ROLE` | `type`、`worker_id`、`role` |
| `ADD_EDGE` | `type`、`source`、`target` |
| `STOP` | `type` |

依次向 `worker_0` 至 `worker_3` 分配角色，随后添加零条或多条边，最后执行 STOP。
角色取值为 `Solver`、`Decomposer`、`Checker`、`Reviser`，允许重复。
边允许任意方向，必须无环；支持空边集、孤立节点和多个 sink。
finalizer 是独立执行组件，不属于 worker graph。

完整合法示例见 `examples/graph/valid.json`，其中包含反向编号边、重复角色和孤立节点。
非法示例见 `examples/graph/invalid.json`，预期错误为步骤索引 1 的 `action_phase`。
步骤索引从零开始；缺少 STOP 时索引为 steps 长度。

JSON 通过标准库解析，结构通过 JSON Schema 校验，图约束通过状态机和 NetworkX 检查。
Markdown 包裹、重复 JSON 键、未知字段和非法动作均被拒绝。Schema 描述结构，
角色分配顺序、重复动作及 DAG 合法性由重放校验；单独通过 schema 不代表合法图。

完整 JSON 解析成功后，按原顺序逐步校验并重放；首个失败步骤之前的快照用于诊断。
JSON 或外层结构无效时不提供动作前缀。任何失败的 `final_graph` 均为 null。
原始文本及边添加顺序均保留，快照不可变；JSON 中每个有效动作（包括 STOP）对应一份快照。
未返回原生 reasoning 时记录 null，空字符串保持为空字符串。

Python 接口：

```python
from pathlib import Path
from ocop.graph import replay

result = replay(Path("examples/graph/valid.json").read_text(encoding="utf-8"))
assert result.valid
assert result.final_graph is not None
```

## 后续实验配置

`config/prototype.json` 保存已确认的实验起始配置，服务 smoke 已接入配置校验。
StrongLLM 使用 DeepSeek 官方 `deepseek-flash`，显式启用 thinking、reasoning_effort=high、top_p=1，temperature=null。
Worker/finalizer 使用 OpenRouter `openai/gpt-4o-mini`，固定 OpenAI provider、禁用 provider fallback，并要求支持所传参数。
密钥通过 python-dotenv 读取或由同名环境变量覆盖，分别使用 `DEEPSEEK_API_KEY` 和 `OPENAI_API_KEY`。
实验输出使用 `artifacts/`；权重使用 `/data1/zhilingyu/models/`。
本地研究依据为 `my_docs/design/design.md` 与 `my_docs/design/Implement-design.md`。

## 服务验证与恢复

```bash
uv run --locked ocop smoke --config config/prototype.json --run-id my-service-check
uv run --locked ocop runs show artifacts/runs/my-service-check
uv run --locked ocop runs export artifacts/runs/my-service-check --output artifacts/service-records.jsonl
```

Smoke 包含两次模型清单查询、一次真实组织提议、一次 worker 请求和一次 finalizer 请求。
使用独立的简单算术题，标记为 smoke split。此处验证服务接口；完整四 worker 调度在公共 executor 中实现。
StrongLLM 通过要求为正常结束、返回原生 reasoning 且产生合法图；worker/finalizer 检查正常返回。
公共 executor 对最终答案进行严格判分，采集模块汇总重复执行标签。

每个 run 目录包含 `records.sqlite3`、内容寻址的 `blobs/`、配置快照 `config.json`、`report.json` 和 `records.jsonl`。
SQLite 外键关联记录及父记录，request 关联 owner，attempt 关联 request；文件使用 SHA-256 校验。
请求先写入 in_flight attempt，原始响应持久化后才解析。usage 缺失保持 null。
只保存允许的响应头；密钥不进入请求快照，响应中意外回显的已知密钥会脱敏。

每个 run 同时只允许一个写进程，进程内支持有界异步请求。恢复同一 run 必须匹配配置与构图契约 hash。
已完成或不完整的响应直接复用；response_saved 可在本地继续解析；中断且无落盘响应的 attempt 记为 uncertain。
uncertain 仍占原请求的三次 attempt 预算，远端可能已执行过该请求，因此不提供跨 API exactly-once 保证。
正常但内容错误/格式错误、length 截断等不会触发请求重试。
网络、429、5xx 等错误有限重试；认证/配置类错误终止 run。达到连续失败阈值也终止 run。
已终止 run 保留现场，修正原因后用新 run ID 启动。

恢复已有验证时使用该 run 自己的配置：

```bash
uv run --locked ocop smoke --config artifacts/runs/my-service-check/config.json --run-id my-service-check
```

`RunStore.put_record` 提供不可变记录和父记录关联，供后续任务、轨迹、执行、标签、SFT 样本及 checkpoint 模块复用。
`RequestRunner` 由一次运行共享，以保证全局并发上限与重复逻辑请求互斥。
SQLite 与 blobs 是权威记录，JSONL/报告可重新导出。当前调用明确使用非流式 Chat Completions。

## 公共图执行与判分

```bash
uv run --locked ocop execute --task examples/execution/task.json --trajectory examples/graph/valid.json --run-id executor-check --repeat-id 0
uv run --locked ocop execute --task examples/execution/task.json --trajectory examples/graph/valid.json --run-id executor-check --repeat-id 0
uv run --locked ocop execute --task examples/execution/task.json --trajectory examples/graph/valid.json --run-id executor-check --repeat-id 1
```

任务 JSON 包含 `task_id`、`question`、字符串 `reference_answer`，可选 `split` 默认为 `smoke`。
输入轨迹必须完整通过当前构图契约重放。四个 worker 按 DAG 依赖单次执行；就绪节点可并行，
worker 输入仅含原题、自身角色 prompt 和按 ID 排序的直接前驱完整输出。
全部 worker 完成后，独立 finalizer 接收原题与四个有序输出；参考答案只用于独立 scorer。

finalizer 最后一行必须为 `#### 数值`，其中数值匹配 `[+-]?[0-9]+(?:\.[0-9]+)?`。
允许末尾 LF/CRLF 换行，末行不能有额外空格或文本。通过 Decimal 精确比较；
GSM8K 参考答案单独提取末行并规范化合法千位分隔符。
完整执行的判分为 `success`、`wrong_answer` 或 `format_error`，后两者记任务失败。
截断等记 `incomplete`，请求重试耗尽记 `infra_failed`，认证/配置或响应模型/provider 不匹配记 `fatal`。
worker 失败后停止调度新节点，收集在途请求结果；节点详情与未调度节点保存在报告中。

`execute` 退出码 0 表示完整执行并完成判分，包括答案错误和格式错误；1 表示执行不完整或服务失败；
2 表示输入、配置或存储冲突。报告中 `status` 与 `score` 分别描述执行状态与任务结果。
相同 run/repeat 恢复已持久化输出，新的 repeat 独立调用四个 worker 和 finalizer。
同一任务在本命令的 run 中绑定一个手写轨迹；更换轨迹时使用新 run ID。

`ocop.executor.execute_repeat` 是采集和评估共用的异步入口，通过 `parents` 关联任务、轨迹和图，
调用方提供共享 `RequestRunner`。接口不接收参考答案，返回执行记录 ID、状态、逐节点结果、最终输出及 usage。
执行配置和 SHA-256 覆盖模型、角色 prompts、消息格式、采样、请求限制、finalizer 与判分/失败规则。
每个节点的请求和所有 attempt 分别归属 worker/finalizer；`known_usage_sum` 仅汇总已知数值，
`unknown_usage_attempts` 单列缺失数量，全缺失时汇总为 null。并发节点耗时之和不是墙钟运行时间。
`execute` 记录单次判分；`collect` 聚合五次完整执行的标签。

真实手写图验收脚本固定执行两个 repeat，并在中间重新打开 run 检查恢复：

```bash
uv run --locked python scripts/verify_executor.py --run-id executor-check
```

正常共 10 次模型调用；失败时保留报告并停止，不自动增加 repeat。
结果写入 run 内的 `verification.json`，其中包含恢复、新 repeat 隔离和 JSONL 导出一致性的验证结果。
`report.json` 为最近一次执行报告；SQLite 中保留各 repeat 的完整记录。
长时间运行使用 tmux 持久化，API 调用仍受配置中的 timeout、attempt 和熔断限制。

## GSM8K 采集与标签

```bash
uv run --locked --extra local ocop collect --run-id gsm8k-20260920 --manifest-only
uv run --locked --extra local ocop collect --run-id gsm8k-20260920
uv run --locked --extra local ocop collect --config artifacts/runs/gsm8k-20260920/config.json --run-id gsm8k-20260920
```

`--manifest-only` 准备数据、冻结任务与候选槽位并退出，无需 API 凭据。
`collect` 开始或恢复采集；长任务应在 tmux 中运行。退出码 0 表示全部候选槽位达到终态，
包括非法提议和预算用尽的 incomplete 标签；1 表示运行因服务故障等停止；2 表示输入或配置错误。
报告中的 `finished` 与 `all_legal_candidates_labeled` 分别表示槽位处理完成和合法候选标签完整，
没有合法图时后者为 null。

数据通过 Datasets 从 Hugging Face 镜像读取，固定 `openai/gsm8k`、`main`、train split 和 revision。
Hub/Arrow 缓存均位于 `artifacts/datasets/`。加载器使用数据集声明的文件配置并保留库的 split 校验；
返回的 train 数据参与抽样。按源行号顺序，以 Python `random.Random(seed).sample` 无放回选择 30 行，
前 24 行为 train、后六行为 eval。源行号、原始题目/参考答案、规范化答案和 SHA-256 写入 task manifest。
全部选中题目的参考答案在调用模型前校验；恢复直接读取 run 中不可变 manifest。

每题三个独立提议槽位；StrongLLM 只接收原题、版本化角色/组织规则及 JSON Schema，经验为空。
原生 reasoning 与 content 分别保存。正常结束且完整重放合法的提议才进入 executor；
非法、截断、请求耗尽分别保留状态，不追加候选。模型身份或 thinking 字段违反已验证契约时停止 run。
相同任务中角色分配及边集相同的图记录为重复，仍分别执行；边添加顺序与原始轨迹保持原样。

默认两个候选并行，所有构图、worker、finalizer 共享请求并发上限八。
每个候选的 repeat 顺序执行，取得五次完整结果即停止，最多八次。
错误答案和格式错误计为完整执行、成功值为零；截断和基础设施失败只消耗 repeat 预算。
`aggregate_label` 保存成功次数、完整 repeat/执行 ID 和 executor hash；五次完整结果产生 `success_count/5`，
包括零成功标签。预算用尽且不足五次时 mean_outcome 为 null、状态为 incomplete。
中断留下的工作保持待恢复状态，不提前写入终态标签。

run 目录新增 `task-manifest.json` 与 `collection-report.json`。报告按 split 展示候选状态、合法率、
同题重复率、标签分布、执行不完整/基础设施失败率及 proposal/worker/finalizer usage。
合法率分母为该 split 全部候选槽位，重复率分母为合法候选，执行失败率分母为已记录的 repeat。
默认每个候选结束后报告与 JSONL 更新，详细状态和原文持续写入 SQLite/blobs。
同一候选、请求、repeat 与标签的关联用于恢复，已完成结果不新增调用或重复计入 usage。

OpenRouter 明确返回 `in_flight_budget_exhausted` 且来源为 `openrouter_in_flight_budget` 时，
请求按基础设施失败有限重试。相同服务共享 `Retry-After` 等待期限（支持秒数与 HTTP 日期，
缺失或无效时等待 120 秒，单次最多 300 秒），重启继承已保存期限；每个请求仍最多三个 attempt。
已因这类响应停止的采集可显式恢复：

```bash
uv run --locked --extra local ocop collect --config artifacts/runs/gsm8k-20260920/config.json --run-id gsm8k-20260920 --resume-inflight-budget
```

恢复核验所有 fatal 请求的原始响应，并写入不可变 recovery 事件后解除停止状态。
冻结配置、历史请求与标签保持原值；失败 repeat 继续占用八次总预算，恢复后补采新的 repeat。
其他致命响应不适用此入口。重试修复由 session 中的代码版本及 recovery 策略版本记录。

充值后，可使用同一命令并将恢复参数换为 `--resume-after-topup`。
此入口核验 OpenRouter `openrouter_credits` 的 HTTP 402 及历史在途预算错误，
记录 `openrouter_topup.v1` 恢复事件；累计 attempt/repeat 预算和已完成记录继续保留。

只读验收工具从 SQLite/blobs 重放合法图、重算判分与标签，并检查槽位、split 关联和请求预算：

```bash
uv run --locked --extra local python scripts/verify_collection.py artifacts/runs/gsm8k-20260920 --output artifacts/gsm8k-20260920-verification.json
```

数据准备期间可增加 `--allow-pending` 检查已保存记录；此时 `passed=true` 只代表当前记录通过检查，
完整采集还要求 `finished=true`。本步骤不构造 SFT 样本，也不启动训练。

## 扩量采集与组织独立复测

`config/gsm8k-expansion.json` 固定新增200道训练题、50道留出题和seed=20260921，
从原采集的冻结GSM8K版本抽样并排除旧30题。留出题仅保存manifest，采集队列不包含这些题。
每道训练题6次独立自由提议，每个合法候选5次完整执行、最多8次repeat；重复图照常独立执行。
worker、角色prompts、finalizer及采样沿用源采集的executor配置。

同一进程并行运行采集与开发题复测，共享总请求并发8及服务等待期限，各流程候选并发2。
开发题217/4597各比较D/S/C/R链与相同角色空边图，每组10次完整执行、最多16次repeat。
采集全部结束并通过验收后，按同题同图汇总完整候选的初测结果，取最大正差值的最多20题；
各固定一对高/低初测组织，独立复测每图20次、最多32次repeat。初测与复测记录分别保存。
每个复测区组的随机顺序先持久化，区组之间等待完成；失败补采占用既定总预算。

```bash
env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training --extra analysis python scripts/fixed_graph_baseline.py --experiment-config config/gsm8k-expansion.json --run-id gsm8k-20260921-expansion --prepare-only
env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training --extra analysis python scripts/fixed_graph_baseline.py --experiment-config config/gsm8k-expansion.json --run-id gsm8k-20260921-expansion
env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training --extra analysis python scripts/fixed_graph_baseline.py --verify-experiment artifacts/experiments/gsm8k-20260921-expansion
```

长实验在tmux中启动。再次执行同一实验命令恢复已保存工作；充值恢复可显式附加
`--resume-after-topup`，适用范围与采集命令一致。冻结配置或选图变化会拒绝恢复。

实验入口默认启用自动恢复，运行策略位于`config/auto-recovery.json`，可用`--recovery-config`指定。
连续10个请求耗尽重试后暂停请求，等待120秒，再探测受影响服务的模型目录。
连接、读取、超时、HTTP 408/429/5xx故障可自动恢复；探测失败时依次等待240、480秒，随后每900秒重试。
探测成功且历史数据校验通过后，自动继续剩余采集和复测，保留全部历史记录与累计attempt/repeat预算。
每次熔断最多96次探测；认证、余额、数据一致性错误或探测预算耗尽时停止并记录原因。
策略、等待时间及探测结果持久化，重启沿用已有状态。恢复事件与解除熔断在同一数据库事务中提交。
监督进程支持挂接已有实验，等待现有写进程退出后接管；文件锁防止重复运行。

单独进行人工连接恢复时，也可使用已有脚本：

```bash
env -u LD_LIBRARY_PATH uv run --locked --extra local --extra training --extra analysis python scripts/resume_evaluation.py --run artifacts/runs/gsm8k-20260921-expansion-collection --recover-only
```

该命令核验原始失败attempt、记录一致性及受影响服务的模型目录（包括DeepSeek和OpenRouter），写入恢复事件后解除熔断，
保留已完成结果和累计预算；HTTP错误或未完成的请求attempt会被拒绝。
随后在tmux中再次执行同一实验命令，继续剩余采集及自动复测。

实验总目录为`artifacts/experiments/<run-id>/`：`experiment.json`保存配置，
`status.json`表示运行阶段，`selection.json`保存独立复测来源，`coverage.json`报告组织覆盖，
`verification.json`保存重算验收与配对统计，`tensorboard/`保存进度指标。
`supervision-policy.json`冻结运行监督策略，`supervisor-state.json`表示等待、冷却、运行或停止状态；
监督策略独立于冻结的研究配置和executor hash，逐次探测证据保存在对应run的数据库中。
三个独立run分别使用`-collection`、`-development`、`-replication`后缀。
扩量采集每25个候选更新报告，结束或中断时导出完整JSONL；逐请求记录持续落盘。

复测按完整配对区组计算10,000次配对bootstrap的95%百分位区间，报告精确McNemar检验；
新数据各题的检验采用Holm校正。单边缺失区组和incomplete另列，不追加样本直到显著。
bootstrap退化区间仅反映实测样本，不能视为总体无不确定性。
开发题的逐节点原文汇总在`development-audit.json`，错误出现、纠正及finalizer取舍需要语义审计；
自动报告将其标记为`pending_manual_review`。本轮不自动开始训练。

## 本地模型环境检查

本地依赖通过独立的 `local` extra 锁定，使用 CUDA 12.1 的 PyTorch 2.5.1、匹配 torchvision 和 Transformers 5.3.0。
PyTorch wheel 使用阿里云镜像；具体版本与下载地址记录于 `uv.lock`。
本服务器的 `LD_LIBRARY_PATH` 包含 CUDA 12.0，运行项目模型检查时对该进程清除该变量：

```bash
uv sync --locked --extra local
env -u LD_LIBRARY_PATH uv run --locked --extra local python scripts/inspect_local_model.py --model-path /data1/zhilingyu/models/Qwen3.5-2B --output artifacts/environment/local-model.json --decode
```

脚本检查本地文件与权重 hash、tokenizer、thinking 模板、CUDA/BF16，并通过 `--decode` 执行最多 128 个新 token 的独立生成。
诊断生成使用单卡、贪心解码和 tokenizer 的消息结束 token；配置、输出及是否自然终止均保存到报告。
这是环境 smoke，正式训练前后评估仍使用 prototype 配置中的生成参数。

2026-09-20 验证记录：87 项测试通过；真实服务 run `services-20260919` 恢复后仍为
5 个请求、5 个 attempt。新环境的权重加载、thinking 模板及短解码检查通过，报告在
`artifacts/environment/local-model-verified.json`。该次生成达到 128 token 上限，未自然结束，
峰值显存 4,464,998,912 bytes。完整本地服务与训练链路在后续步骤验证。
脚本使用 `return_dict=False` 显式取得 chat template 的 token 列表。

服务接口依据：[DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)、
[OpenRouter API](https://openrouter.ai/docs/api/reference/overview)。
