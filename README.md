# AI Art Curator

**AI Art Museum — Curated by Human, Created by AI**

一个由私人设备调度、由人批准、由静态网站展出的 AI 艺术馆。默认后端为 FLUX 3 Image Klein，生产生图使用 HPC 单张 A100；每天最多一次任务、每次最多 30 分钟，不自动回退到本地 GPU。生产清单从零件作品开始。模拟候选、原始邮件、编号、审批记录和模型权重各自保存在私有目录。

## 五分钟跑通本地模拟

需要 Python 3.12、Node.js 20.19+ 或 22、npm、Bash。模拟生图使用 Pillow，不需要 GPU、Gmail、GitHub 或学校账户。

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
bash config/demo.sh
(cd web && npm ci && npm run build)
(cd web && ART_DEMO_PUBLIC=../work/demo/public npm run dev -- --host 127.0.0.1)
```

打开终端显示的本地地址。模拟脚本生成同日 8 张候选，写入独立 HTML 邮件预览，模拟批准 01–04，并把 4 张批准作品放在 `work/demo/public/gallery/`。在 `work/demo/private/mail/` 可以查看 HTML、MIME 发件箱和模拟回信。生产构建只读取 `web/public/`，并拒绝模拟作品或未列入清单的公开图片。

逐步执行同一流程时，在仓库根目录使用以下命令。

```bash
export ART_BACKEND=mock
export ART_STATE_DIR=work/manual/private ART_PUBLIC_DIR=work/manual/public
python -m artcurator generate --count 8
python -m artcurator preview-email --latest
python -m artcurator send-email --latest --mock
python -m artcurator mock-reply --latest --text 'APPROVE: 01, 03, 04'
python -m artcurator check-replies --mock
python -m artcurator publish --approved-only --mock
python -m artcurator scheduler --dry-run
python -m artcurator status
```

`--latest` 选择最新的可投递批次，也可以传 `--day YYYY-MM-DD`。每封邮件的第一个有效 `APPROVE: 01, 03`、`APPROVE: ALL` 或 `SKIP` 会结束该轮审批。批次不完整时编号可能跳号。重复调用发送、审批和发布保持幂等。CLI 输出空结果、跳过、部分成功或失败的状态。普通错误返回 1，无 GPU 被调度器记录为正常跳过。

## HPC A100 模型环境

目标为经授权的 HPC A100 40GB。请按管理员允许的方式准备 Linux、Python 3.12 和兼容集群驱动的 PyTorch/CUDA 环境；以下命令仅准备依赖和缓存，不申请或启动 GPU。避免继承模拟示例的 `ART_*` 导出。

```bash
# 私人调度设备的 .env 不得复制到 HPC。
python3.12 -m venv .venv-gpu
. .venv-gpu/bin/activate
python -m pip install -e . -r requirements-flux.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130
hf auth login
hf download black-forest-labs/flux-3-image-klein \
  --local-dir weights/flux-3-image-klein \
  --include model.safetensors ae.safetensors 'text_encoder/*'
hf download black-forest-labs/flux-3-image-klein-pup --local-dir weights/pup
# 在私人调度设备检查计划（不会提交）：
python -m artcurator scheduler --dry-run
```

先在 Hugging Face 账号中阅读并同意 gated 模型条款。FLUX 3 Image Klein 是开放权重模型，使用 **FLUX Kommunity License**，其上采样器使用 Apache-2.0。私人非商业生图及公开展示作品仍须按模型条款和使用政策操作。官方仓库描述模型组合约占 52 GB，可使用 `ART_WEIGHTS_DIR` 指向预缓存目录。推理时强制离线并预检权重，下载单独执行。私有调度器可使用轻量 `.venv`，在本机 `.env` 将 `HPC_REMOTE_PYTHON` 指向远端 GPU 环境解释器的绝对路径。[官方实现](https://github.com/black-forest-labs/flux-3-image-klein) · [FLUX 权重和条款](https://huggingface.co/black-forest-labs/flux-3-image-klein) · [Prompt Upsampler](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup)

默认 16-step distilled、原生 BF16、无 CFG、方形 1024×1024。可在 `generation.aspect_ratios` 混用 `1:1`、`16:9`（1280×720）、`3:2`（1248×832）、`9:16`（720×1280）。先单独启动官方 vLLM Upsampler，把每个扩展 Prompt 持久化。退出该阶段后，默认将官方文本编码器、DiT、VAE 分别迁入 GPU，使用官方采样及解码函数，降低同时驻留的显存。A100 路径没有 FP8 或 NVFP4 开关。官方 CUDA 依赖的实际可安装性和显存峰值须在获授权的 A100 作业内验证。[BFL 的显存与分阶段说明](https://github.com/black-forest-labs/flux-3-image-klein#smaller-gpus)

**备用模型。** 在独立 GPU venv 中，先按目标驱动安装支持对应架构的 CUDA PyTorch，再运行以下命令。在私人调度设备的 `.env` 设置 `ART_BACKEND=diffusers`，并让 `HPC_REMOTE_PYTHON` 指向准备好的远端环境。默认使用官方 Z-Image-Turbo（Apache-2.0）及 Diffusers 0.39.0，9 steps、guidance 0。其他 Hugging Face Diffusers 文生图模型须支持已使用的 pipeline 参数。核验许可后，配置 `diffusers_model`、`diffusers_steps`、`diffusers_guidance` 并打开 `diffusers_license_approved`，然后以相同的目录约定预下载。批次的生成配置在建档时冻结，切换模型可用于下一批。[Z-Image-Turbo 模型卡](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) · [Diffusers 版本](https://github.com/huggingface/diffusers/releases/tag/v0.39.0)

```bash
python -m pip install -e . -r requirements-diffusers.txt
hf download Tongyi-MAI/Z-Image-Turbo --local-dir weights/diffusers/Tongyi-MAI/Z-Image-Turbo
```

## Caviness 与 Slurm

`HPC_ENABLED`、`HPC_AUTHORIZED` 和 `HPC_OPPORTUNISTIC` 都默认关闭。先向管理员确认校园账号及用途授权、可抢占或低优先级机制、允许的账户、分区、QoS、A100 40GB GRES、文件系统和驱动。默认 `scheduler.mode: slurm`。若无合适配置，保持 HPC 开关关闭，或设为 `"off"`；不会回退到 RTX 5090 或其他本地 GPU。脚本通过 Slurm 正式申请资源。[UD Slurm 培训资料](https://docs.hpc.udel.edu/_media/training/slurm/introduction_to_slurm_october2024.pdf) · [Slurm sbatch 文档](https://slurm.schedmd.com/sbatch.html)

1. 在私人设备运行 `python -m artcurator scheduler --dry-run`。此命令仅打印计划。
2. 取得授权后，把代码、环境及已许可的权重放到管理员允许的位置。远端只需要 Python 基础包与模型环境。创建私有 `private/` 和 `weights/`，保持 OAuth、Gmail 和 Git 发布凭据在私人设备。SQLite 与文件锁需要支持 POSIX 锁的持久文件系统。
3. 已根据报告的 Slurm 17.11.8 调整参数：使用 `--export=PATH=/usr/bin:/bin` 显式传递最小环境（该版本不支持 `NIL`），通过远端 `id -u` 和 `squeue --user` 查询本账号作业（不使用 `--me`）。保留 `--deadline`、`--no-requeue`。让管理员核实 `gpu:a100:1` 名称、A100 40GB 可见范围及分区/QoS 权限。Slurm 解析提交时间时使用提交主机时区，当前配置应与该主机一致。先核对 SSH host key，配置管理员允许的非交互式 SSH 路径。参数参照 [Slurm 17.11.8 sbatch 手册](https://github.com/SchedMD/slurm/blob/slurm-17-11-8-1/doc/man/man1/sbatch.1)和 [squeue 手册](https://github.com/SchedMD/slurm/blob/slurm-17-11-8-1/doc/man/man1/squeue.1)；尚未完成真实集群作业验证。
4. 在私机填入 `HPC_HOST`、`HPC_ACCOUNT`、`HPC_PARTITION`、`HPC_QOS`、`HPC_REMOTE_ROOT`、`HPC_REMOTE_PYTHON`。路径用远端绝对路径。确认后在私机设置三个 HPC 开关，保持 `scheduler.mode: slurm`，再使用 `scheduler` 或 `tick`。默认时间上限 `runtime_minutes: 30` 不能超过 30；作业为 `--gres=gpu:a100:1 --time=00:30:00`。

每个本地日期共享一条 SQLite 调度记录，每天最多调用一次真实 `sbatch`，累计 GPU 运行时间不超过 30 分钟。默认请求一张 A100、30 分钟墙钟和约 10 分钟排队准入窗口。若接近午夜，调度器收紧窗口。等待资源失败时当日作废，并在后续状态检查中取消超期作业。抢占后不自动重新入队。私机发出短时 SSH 命令记录状态，并在作业退出后拉取已完成的候选。无需在登录节点保持轮询进程。

**释放 GPU。** 正常完成、超时、抢占或异常退出时，worker 会保存已完成作品、丢弃模型引用并尽力释放本进程的 PyTorch 空闲缓存。Prompt Upsampler 使用独立进程组，父进程在所有退出路径回收该组，避免残留 vLLM 子进程。worker 退出释放 CUDA context，Slurm 负责收回作业分配；强制终止时由 Slurm 清理作业进程。不会重置 GPU、终止其他用户进程或删除权重缓存。`empty_cache()` 只释放未使用的缓存，不能替代进程退出，也不保证共享节点的总显存使用率归零。此清理路径有模拟 CUDA 与真实 CPU 子进程测试，尚未完成真实 A100 验证。

**恢复。** 按图保存的检查点包括 Prompt、PNG、种子和原始生成时间。恢复从未完成编号继续，单张推理中间的扩散状态会重新计算。默认将抢占时已保存的部分候选送审。使用分开的 `scheduler`、`send-email` 命令可以先保留未送审批次，在之后一个日期的唯一配额内执行 `python -m artcurator scheduler --resume YYYY-MM-DD` 恢复 Slurm 批次，或用 `python -m artcurator generate --resume --day YYYY-MM-DD` 恢复本地批次。已投递批次的编号和内容冻结。主机中断导致提交结果不明时，按唯一 JobName 对账。无资源而跳过的日批次不会自动补跑。

生产模式请使用 `scheduler` 或 `tick`，不要在 login node 上执行 `generate`。`generate` 的本地真实生图仅在显式设置 `scheduler.mode: local` 后可用；本地 mock 演示仍可运行。

## Gmail OAuth 和审批

在私人设备上运行 `python -m pip install -e '.[gmail]'`。使用专用 Gmail 发件账号，避免让 OAuth 访问日常主邮箱。发件账号和批准账号必须是不同邮箱。

1. 在 Google Cloud 项目启用 Gmail API。配置 OAuth consent screen 并按 Workspace 政策添加测试用户或选择内部应用。
2. 创建 **Desktop app** OAuth client。保存 JSON 到 `private/oauth-client.json`，仅私有设备可读。在本机 `.env` 配置 `GMAIL_ENABLED=true`、`GMAIL_SENDER`、`GMAIL_RECIPIENT`、`GMAIL_APPROVER`。最后一项省略时等于收件地址。可显式指定 `GMAIL_CLIENT_SECRET` 和 `GMAIL_TOKEN`。
3. 在可访问本机浏览器的环境中运行 `python -m artcurator gmail-auth`，选择发件账号授权。OAuth 回调绑定本机，刷新令牌私下落盘。外部测试应用可能需要按 Google 的测试策略定期重新授权。
4. 先检查 `python -m artcurator preview-email --latest`。在批准真实发送后，运行 `python -m artcurator send-email --latest`。用已配置的发件人地址直接回复收到的原始会话，独立一行写 `APPROVE: 01, 03, 04`、`APPROVE: ALL` 或 `SKIP`。然后运行 `python -m artcurator check-replies`。

请求范围为 `gmail.send` 和 `gmail.readonly`。后者读取指定会话正文以及在 Sent 中对账。审批检查精确 From、Gmail 的 DMARC 对齐、会话 ID、原始 Message-ID 中的批次和随机值，以及 In-Reply-To、References 和编号。邮件引用和转发不触发批准。默认只读取过去 30 天仍未决定的已发送会话。已明确拒绝的发送有次数上限。网络失败导致接受状态不明时，下次运行只查找 Sent，等待确认以避免重复发送。可用 `status` 查看结果。首次真实联调需要验证选用邮箱的 DMARC 头和回复客户端保留的引用头。[Gmail API scopes](https://developers.google.com/workspace/gmail/api/auth/scopes) · [Gmail threads](https://developers.google.com/workspace/gmail/api/guides/threads)

## GitHub Pages 与定时执行

在你准备公开画廊时，手动创建或选定有 GitHub Pages 权限的仓库。该版本要求发布分支为 `main`，更改时需要同时修改 `config.yaml` 和工作流分支过滤条件。项目初始工作目录没有 Git 历史，若将代码放进已有仓库，保留该仓库的历史。

1. 检查并提交项目源文件和初始空清单。保持 `private/`、`work/`、`.env`、权重及 OAuth 数据在 Git 之外。
2. 在 GitHub 的 **Settings → Pages → Build and deployment** 选择 **GitHub Actions**。由你授权并完成首次 push。工作流在 Ubuntu 上执行 Python 测试、TypeScript 和 Vite 构建、桌面和移动 Chromium 测试。通过后将 `web/dist` 交给 Pages。作品详情使用 hash 路由，兼容 Pages 子路径。[GitHub Pages 自定义工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
3. 在私机设置专用发布 checkout，配置限于此仓库的 Git 凭据、Git 提交身份和 `origin`。用 `python -m artcurator publish --approved-only` 先检查本地发布清单。授权自动推送后在私机设置 `GIT_PUSH_ENABLED=true`。此时 `python -m artcurator publish --approved-only --push` 仅分阶段提交批准原图、缩略图和清单，并推送分支。远端领先、暂存区有文件或本地存在无关提交时需要先人工处理。
4. 安装 `config/cron.example` 中的私机定时任务。修改绝对路径后加入私机 crontab。先运行一次 `python -m artcurator tick` 做已授权的端到端验收。它分别处理每日生成、Gmail 投递及检查、清单更新与可选推送。也可分别调度子命令。macOS 可用 launchd；Windows 建议在 WSL2 内调度。

本地存储适配器保存不可变 PNG 和 WebP 缩略图，最后原子替换 manifest。失败后重跑会补齐同一作品的缺失文件。CI 检查 manifest 与公开的每个图片是否一一对应。随着库增长，可以按 `FilesystemGallery.put()` 的契约引入对象存储并同步调整构建校验及前端资源路径。MVP 使用 GitHub Pages 的静态文件。

## 策展与偏好

内置 12 个主题库、12 种色彩组合和构图候选，避开上一批所用的主题与配色组合。SQLite 保存作品和编号、Prompt、种子、分辨率、模型等字段。生成端使用 SHA-256 和 dHash 去重，并剔除异常尺寸、过暗、过亮或低对比度图片。可选 `aesthetic_score` 是基于对比度及亮度的可解释启发式，仅作记录。人类审批决定作品能否发布。

偏好引擎用最近的批准和曝光记录计算主题、色彩、构图及风格权重，并轻量提高已批准 Prompt 关键词的权重。审批快照也记录模型、Seed 和时间。模型选择由配置决定。默认保留 30% 的随机探索。画廊支持主题和月份筛选、Masonry、详情、原图、下载及浏览器内收藏。每月精选以当月最新批准的作品为基础，本地收藏优先。

## 验证

```bash
python -m pytest
python -m artcurator scheduler --dry-run
bash -n config/slurm.sbatch
(cd web && npm ci && npm run build)
(cd web && npx playwright install chromium && npm run test:e2e)
```

测试覆盖设备不可用、同日去重、Slurm 授权和超时、抢占后同步与恢复、MIME、错误邮箱和会话、批次隔离、回复和发布幂等，以及本地裸 Git 仓库的提交与推送。Playwright 使用隔离的模拟产物和生产空清单测试桌面与手机。真实 CUDA、生图、Caviness 提交、Gmail OAuth 发送和 GitHub Pages 云端部署都需要目标环境验收。

## 项目文件结构

```text
.
├── .env.example
├── .gitignore
├── .github/workflows/pages.yml
├── README.md
├── config.yaml
├── pyproject.toml
├── requirements-flux.txt
├── requirements-diffusers.txt
├── config/
│   ├── cron.example
│   ├── demo.sh
│   ├── demo.yaml
│   ├── e2e.sh
│   └── slurm.sbatch
├── artcurator/
│   ├── __init__.py
│   ├── __main__.py
│   ├── settings.py
│   ├── store.py
│   ├── curator/{__init__,publish}.py
│   ├── generator/{__init__,backends,flux,plan,worker}.py
│   ├── mailer/{__init__,format,replies,service,transport}.py
│   ├── preference_engine/{__init__,engine}.py
│   └── scheduler/{__init__,service,slurm}.py
├── tests/{conftest,test_generation_scheduler,test_mail_publish}.py
└── web/
    ├── e2e/museum.spec.ts
    ├── index.html
    ├── package.json
    ├── package-lock.json
    ├── playwright.config.ts
    ├── public/gallery/manifest.json
    ├── scripts/check-gallery.mjs
    ├── src/{gallery.ts,main.tsx,style.css}
    ├── tsconfig.json
    └── vite.config.ts
```
