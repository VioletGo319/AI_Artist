# AI Art Curator

**AI Art Museum — Curated by Human, Created by AI**

A private AI art museum with generation scheduled from a personal device, human approval by email, and a static public gallery. The default model backend is FLUX 3 Image Klein. Production generation uses one authorized HPC A100 GPU, with at most one job per day and a maximum runtime of 30 minutes. There is no automatic fallback to a local GPU.

The public gallery starts empty. Candidate images, raw email, batch identifiers, approval records, and model weights are stored privately. Only explicitly approved artwork is published.

## Quick start: local mock workflow

Requirements: Python 3.12, Node.js 20.19+ or 22, npm, and Bash. The mock generator uses Pillow and requires no GPU, Gmail access, GitHub access, or university account.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
bash config/demo.sh
(cd web && npm ci && npm run build)
(cd web && ART_DEMO_PUBLIC=../work/demo/public npm run dev -- --host 127.0.0.1)
```

Open the local URL printed in the terminal. The demo generates eight candidates for one day, creates an HTML email preview, simulates approval of images 01–04, and writes those four images to `work/demo/public/gallery/`. The HTML preview, MIME outbox, and simulated replies are available under `work/demo/private/mail/`.

Production builds read only `web/public/` and reject mock artwork or public image files missing from the gallery manifest.

To run the same workflow step by step, execute these commands from the repository root:

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

`--latest` selects the latest batch ready for delivery. Use `--day YYYY-MM-DD` to select a specific date. The first valid `APPROVE: 01, 03`, `APPROVE: ALL`, or `SKIP` reply closes the batch's approval round. Incomplete batches may have gaps in their image numbers.

Sending, approval processing, and publication are idempotent. CLI commands report empty, skipped, partial, or failed outcomes. Ordinary errors return exit code 1; the scheduler records unavailable GPUs as a normal skip.

## HPC A100 model environment

The original target is an authorized HPC A100 40GB. Prepare Linux, Python 3.12, and a PyTorch/CUDA environment compatible with the cluster's driver using an administrator-approved method. The commands below install dependencies and cache weights; they do not request a GPU or start inference. Clear the mock demo's `ART_*` environment overrides before configuring production.

**Check driver compatibility first.** The FLUX installation below follows the official CUDA 13 package source. For Caviness driver `550.54.15`, use the validated CUDA 12.4 Z-Image-Turbo setup described in the next section instead.

```bash
# Never copy the private scheduler's .env file to the HPC cluster.
python3.12 -m venv .venv-gpu
. .venv-gpu/bin/activate
python -m pip install -e . -r requirements-flux.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130
hf auth login
hf download black-forest-labs/flux-3-image-klein \
  --local-dir weights/flux-3-image-klein \
  --include model.safetensors ae.safetensors 'text_encoder/*'
hf download black-forest-labs/flux-3-image-klein-pup --local-dir weights/pup
# Inspect the plan on the private scheduler device without submitting a job:
python -m artcurator scheduler --dry-run
```

Read and accept the gated model's terms using your Hugging Face account before downloading. FLUX 3 Image Klein is an open-weight model under the **FLUX Kommunity License**; its Prompt Upsampler uses Apache-2.0. Private noncommercial generation and public display of outputs remain subject to the applicable license and usage policy.

The official repository describes a combined model footprint of approximately 52 GB. Set `ART_WEIGHTS_DIR` to an existing cache if needed. Inference checks the cache and runs offline; downloads are a separate setup step. The private scheduler can use a lightweight `.venv`. Set `HPC_REMOTE_PYTHON` in its local `.env` to the absolute path of the remote GPU interpreter or container wrapper. [Official implementation](https://github.com/black-forest-labs/flux-3-image-klein) · [FLUX weights and terms](https://huggingface.co/black-forest-labs/flux-3-image-klein) · [Prompt Upsampler](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup)

FLUX defaults to 16-step distilled inference, native BF16, no CFG, and 1024×1024 output. Configure `generation.aspect_ratios` to use `1:1`, `16:9` (1280×720), `3:2` (1248×832), or `9:16` (720×1280).

The official vLLM Upsampler runs in a separate stage and saves each expanded prompt. After that process exits, the default inference path moves the official text encoder, DiT, and VAE onto the GPU separately, using the official sampling and decoding functions to reduce simultaneous GPU residency. The A100 path does not enable FP8 or NVFP4. Dependency compatibility and peak GPU memory must still be verified inside an authorized A100 allocation. [BFL memory and staged-inference guidance](https://github.com/black-forest-labs/flux-3-image-klein#smaller-gpus)

**Fallback model.** In a separate GPU virtual environment, first install a CUDA-enabled PyTorch build compatible with the target driver and GPU architecture, then run the commands below. Set `ART_BACKEND=diffusers` on the private scheduler and point `HPC_REMOTE_PYTHON` to the prepared remote environment.

The default fallback is the official Z-Image-Turbo model (Apache-2.0), using Diffusers 0.39.0, nine steps, and guidance 0. Other Hugging Face Diffusers text-to-image models must support the pipeline arguments used by this project. After reviewing a replacement model's license, configure `diffusers_model`, `diffusers_steps`, and `diffusers_guidance`, enable `diffusers_license_approved`, and download its weights using the same directory convention. Generation settings are frozen when a batch is created; model changes apply to subsequent batches. [Z-Image-Turbo model card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) · [Diffusers release](https://github.com/huggingface/diffusers/releases/tag/v0.39.0)

```bash
python -m pip install -e . -r requirements-diffusers.txt
hf download Tongyi-MAI/Z-Image-Turbo --local-dir weights/diffusers/Tongyi-MAI/Z-Image-Turbo
```

## Caviness and Slurm

`HPC_ENABLED`, `HPC_AUTHORIZED`, and `HPC_OPPORTUNISTIC` are disabled by default. Confirm account and workload authorization, an approved preemptible or low-priority scheduling mechanism, the account, partition, QoS, A100 GRES, filesystem, and driver with the administrator.

The default is `scheduler.mode: slurm`. If no suitable configuration is available, keep the HPC switches disabled or set the scheduler mode to `"off"`. The application does not fall back to an RTX 5090 or another local GPU. All HPC allocations are requested through Slurm. [UD Slurm training](https://docs.hpc.udel.edu/_media/training/slurm/introduction_to_slurm_october2024.pdf) · [Slurm sbatch documentation](https://slurm.schedmd.com/sbatch.html)

1. Run `python -m artcurator scheduler --dry-run` on the private device. This prints a plan without submitting it.
2. After authorization, place the code, environment, and licensed weights in an approved location. The remote worker needs only the Python base dependencies and model environment. Create private `private/` and `weights/` directories. Keep OAuth, Gmail, and Git publishing credentials on the private device. SQLite and file locks require persistent storage with POSIX locking support.
3. The adapter supports the reported Slurm 17.11.8 installation. It passes a minimal environment using `--export=PATH=/usr/bin:/bin` because that version does not support `NIL`. It resolves the remote user's UID with `id -u` and queries jobs with `squeue --user`, without using `--me`. It retains `--deadline` and `--no-requeue`. Confirm `gpu:a100:1`, available A100 variants, and partition/QoS permissions with the administrator. Slurm interprets submission times in the submission host's timezone; configure the scheduler to match. Verify the SSH host key and configure an approved noninteractive SSH connection. See the [Slurm 17.11.8 sbatch manual](https://github.com/SchedMD/slurm/blob/slurm-17-11-8-1/doc/man/man1/sbatch.1) and [squeue manual](https://github.com/SchedMD/slurm/blob/slurm-17-11-8-1/doc/man/man1/squeue.1). Real Slurm submission and result retrieval have been validated for the Z-Image-Turbo run described below.
4. On the private device, configure `HPC_HOST`, `HPC_ACCOUNT`, `HPC_PARTITION`, `HPC_QOS`, `HPC_REMOTE_ROOT`, and `HPC_REMOTE_PYTHON`. For Caviness, also set `HPC_WORKGROUP` to the authorized workgroup. SSH commands enter that group through `workgroup --command` for submission, queries, and retrieval. Use absolute remote paths. Once the environment is approved and ready, enable all three HPC switches, retain `scheduler.mode: slurm`, and use `scheduler` or `tick`. `runtime_minutes: 30` is the default and cannot exceed 30. Jobs request `--gres=gpu:a100:1 --time=00:30:00`.

### CentOS 7 containers

Many modern Python/CUDA wheels cannot run directly on a host with glibc 2.17. Apptainer separates the lightweight control environment from the GPU model environment:

- Create remote `containers/` and `private/` directories.
- Copy `config/hpc-runtime.env.example` to `private/runtime.env`, with permissions set to 600.
- Configure absolute paths for Apptainer, the SIF images, and their Python interpreters.
- Make `config/hpc-python.sh` executable and point the private scheduler's `HPC_REMOTE_PYTHON` to its absolute remote path.
- Include `config.yaml` in the remote project root.

If compute nodes lack a system `unsquashfs`, set `HPC_UNSQUASHFS` to the administrator-installed shared executable. The current Caviness path is `/opt/shared/squashfs-tools/4.6.1/bin/unsquashfs`. The wrapper adds that directory to Slurm's minimal PATH. A tool present on the login node may be absent from compute nodes; this caused an initial startup failure before model loading and is covered by a regression test.

On login nodes, the wrapper uses the control image and clears `CUDA_VISIBLE_DEVICES`. Inside a Slurm job, it requires an allocated GPU identifier, selects the model environment, enables `--nv`, and preserves Slurm's device mapping. It uses a clean environment, binds the project directory and job scratch when available, and does not pass Gmail/OAuth credentials. Temporary SIF extraction uses node-local storage to avoid NFS cleanup problems.

The official `python:3.12-slim-bookworm` image can host `.venv-control` with `pip install -e .`. Install the pinned model dependencies and cached weights separately for the GPU environment. Passing control-container or mock tests does not prove A100 inference works. The full test suite also requires `git` and `ps`, which are absent from the minimal control image.

A container cannot replace the host GPU driver. Obtain the **actual driver version on the A100 compute node** before selecting CUDA, PyTorch, and vLLM versions. A CUDA toolkit version listed by VALET is not a substitute. Compatibility probes must also use Slurm and count toward the daily attempt limit. Do not install system glibc, run inference, or keep a polling service on a login node. [Apptainer GPU requirements](https://apptainer.org/docs/user/latest/gpu.html)

### Validated CUDA 12.4 fallback

An authorized single-A100 Slurm probe reported driver `550.54.15`. For that environment, use Z-Image-Turbo with CUDA 12.4 rather than the official FLUX default CUDA 13 installation path. Run the following commands inside the container's separate `.venv-zimage`, installing the official PyTorch wheel before the constrained dependencies:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e . -r requirements-diffusers.txt -c config/caviness-zimage-constraints.txt
python -m pip check
```

Dependency checks and `ZImagePipeline` imports passed under Python 3.12. On **November 22, 2026**, a real **single A100 80GB PCIe** generated eight 1024×1024 candidates using BF16, CPU offload, and nine steps. All images were retrieved successfully, and the job and Slurm cleanup steps completed. A100 40GB inference, FLUX inference, and peak GPU memory remain unmeasured. [Official PyTorch versions](https://pytorch.org/get-started/previous-versions/) · [NVIDIA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

Set `ART_BACKEND=diffusers` on the private scheduler and `HPC_GPU_PYTHON` to the remote `.venv-zimage/bin/python`. The validated Python SIF can be reused: the PyTorch wheel supplies the CUDA runtime, while GPU jobs access the host driver through `--nv`. Model, Hugging Face, and CUDA compilation caches remain under the private project directories rather than relying on the container's default home directory.

### Daily limits and resource release

One SQLite schedule record is shared by all attempts for a local calendar date. Normal scheduling invokes real `sbatch` at most once per day, with a maximum cumulative GPU runtime of 30 minutes. The default request is one A100, a 30-minute wall-clock limit, and approximately ten minutes for queue admission. The scheduler shortens the window near midnight, skips an unsuccessful day, and cancels expired jobs during subsequent reconciliation. Preempted jobs are not automatically requeued.

The private device makes short SSH calls to reconcile state and retrieves completed candidates after the job exits. No persistent polling process is needed on the login node.

GPU validation requires exactly one full A100, accepting both 40GB and 80GB variants while rejecting multiple visible GPUs, other models, and MIG slices. Logs report the actual GPU name, memory, and visible device count. The `gpu:a100` GRES alone does not identify memory capacity. Both supported variants use BF16.

**GPU cleanup.** On normal completion, timeout, preemption, or failure, the worker preserves completed images, drops model references, and attempts to release its unused PyTorch GPU cache. The Prompt Upsampler runs in a separate process group; the parent cleans up that group on every exit path to prevent orphaned vLLM workers. Process exit releases the CUDA context, and Slurm reclaims the allocation. Slurm also cleans up job processes after forced termination.

The application does not reset GPUs, terminate other users' processes, or delete cached weights. `empty_cache()` only releases unused cached memory; it does not replace process exit or guarantee that a shared node's total GPU memory usage becomes zero. Cleanup has mock CUDA and real CPU subprocess tests. Following the successful Z-Image-Turbo run on an A100 80GB, the Slurm job and extern cleanup step completed and the job left the queue. Real preemption, forced timeout, and FLUX/vLLM subprocess cleanup still require validation on the target environment.

**Resume.** Per-image checkpoints preserve prompts, PNG files, seeds, and original generation timestamps. Resuming continues from unfinished image numbers; an interrupted image's diffusion steps are recomputed. By default, candidates saved before preemption can be sent for approval.

To retain a partial batch before sending it, run `scheduler` and `send-email` separately. On a later date, use that day's single allocation with `python -m artcurator scheduler --resume YYYY-MM-DD` to resume a Slurm batch, or `python -m artcurator generate --resume --day YYYY-MM-DD` for a local batch. Delivered batches have immutable contents and numbering. If a connection fails after submission and acceptance is uncertain, reconcile using the unique job name. Days skipped for lack of resources are not automatically carried forward.

Use `scheduler` or `tick` for production. Do not run `generate` on a login node. Local real-GPU generation requires an explicit `scheduler.mode: local` configuration; local mock demos remain available.

## Gmail OAuth and human approval

Install the Gmail dependencies on the private device with `python -m pip install -e '.[gmail]'`. Use a dedicated Gmail sending account so OAuth does not need access to your everyday mailbox. The sending and approving accounts must be different addresses.

1. Enable the Gmail API in a Google Cloud project. Configure the OAuth consent screen and add test users or select an internal application according to your Workspace policy.
2. Create a **Desktop app** OAuth client. Save the JSON file as `private/oauth-client.json`, readable only on the private device. Configure `GMAIL_ENABLED=true`, `GMAIL_SENDER`, `GMAIL_RECIPIENT`, and `GMAIL_APPROVER` in the local `.env`. The approver defaults to the recipient if omitted. Override credential paths with `GMAIL_CLIENT_SECRET` and `GMAIL_TOKEN` if needed.
3. Run `python -m artcurator gmail-auth` where a local browser is available and authorize the sending account. The OAuth callback binds locally, and the refresh token is stored privately. External applications in testing mode may need periodic reauthorization under Google's testing policies.
4. Inspect `python -m artcurator preview-email --latest`. After authorizing real sending, run `python -m artcurator send-email --latest`. Reply directly to the original conversation from the configured approving address, using a standalone line containing `APPROVE: 01, 03, 04`, `APPROVE: ALL`, or `SKIP`. Then run `python -m artcurator check-replies`.

The requested scopes are `gmail.send` and `gmail.readonly`. Read access is used to inspect the designated conversations and reconcile messages in Sent. Approval validation checks the exact sender, Gmail's DMARC alignment, thread ID, the batch and random token embedded in the original Message-ID, `In-Reply-To`, `References`, and image numbers. Quoted history and forwarded messages do not trigger approval.

By default, only undecided conversations sent within the last 30 days are checked. Explicit send rejections have bounded retries. If a network failure leaves message acceptance uncertain, the next run searches Sent instead of resending, avoiding duplicate mail. Inspect the outcome with `status`. Real integration testing must verify DMARC headers and the reply client's preservation of threading headers. [Gmail API scopes](https://developers.google.com/workspace/gmail/api/auth/scopes) · [Gmail threads](https://developers.google.com/workspace/gmail/api/guides/threads)

## GitHub Pages and scheduled operation

When ready to publish, create or select a repository with GitHub Pages access. This version uses `main` as the publishing branch. Changing it requires updates to both `config.yaml` and the workflow's branch filters. Preserve existing Git history when adopting the project in another repository.

1. Review and commit the source code and initially empty gallery manifest. Keep `private/`, `work/`, `.env`, weights, and OAuth data out of Git.
2. In GitHub, select **Settings → Pages → Build and deployment → GitHub Actions**. Authorize and perform the initial push. The Ubuntu workflow runs Python tests, TypeScript/Vite builds, and desktop/mobile Chromium tests, then deploys `web/dist` to Pages. Artwork details use hash routing for compatibility with Pages subpaths. [GitHub Pages custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
3. Set up a dedicated publishing checkout on the private device, with repository-scoped Git credentials, a commit identity, and `origin`. Use `python -m artcurator publish --approved-only` to prepare the local public manifest. After authorizing automatic pushes, set `GIT_PUSH_ENABLED=true`. Then `python -m artcurator publish --approved-only --push` stages and commits only approved originals, thumbnails, and the manifest before pushing. A remote branch ahead of the checkout, existing staged changes, or unrelated local commits require manual resolution.
4. Adapt the absolute paths in `config/cron.example` and install the schedule in the private device's crontab. First run `python -m artcurator tick` for an authorized end-to-end check. It handles daily generation, Gmail delivery and reply checks, manifest updates, and optional pushing as separate operations. Individual commands can also be scheduled separately. macOS users can use launchd; WSL2 is recommended on Windows.

The filesystem storage adapter writes immutable PNG originals and WebP thumbnails before atomically replacing the manifest. Rerunning publication repairs missing files for the same artwork without duplicating it. CI verifies that public image files and manifest entries match. If the collection outgrows Git storage, implement the `FilesystemGallery.put()` contract using object storage and update frontend asset paths and build validation accordingly. The MVP uses static files on GitHub Pages.

## Curation and preference learning

The prompt library contains 12 themes, 12 color palettes, and composition variations, avoiding theme/palette combinations from the previous batch. SQLite records artwork IDs and numbers, prompts, seeds, resolutions, models, and other metadata. Generation uses SHA-256 and dHash for deduplication and rejects incorrect dimensions, excessively dark or bright images, and low contrast. Optional `aesthetic_score` values are explainable brightness/contrast heuristics for reference only. Human approval determines publication.

The preference engine uses recent approvals and exposures to weight themes, palettes, compositions, and styles, with a small boost for keywords in approved prompts. Approval snapshots also record the model, seed, and selection time. Model selection remains configuration-driven. The default exploration rate is 30%.

The gallery supports theme and month filters, a masonry layout, artwork details, original images, downloads, and browser-local favorites. Monthly selections are based on the latest approved works for that month, prioritizing local favorites.

## Validation

```bash
python -m pytest
python -m artcurator scheduler --dry-run
bash -n config/slurm.sbatch
(cd web && npm ci && npm run build)
(cd web && npx playwright install chromium && npm run test:e2e)
```

Automated tests cover unavailable devices, daily deduplication, Slurm authorization and deadlines, synchronization and resume after preemption, MIME generation, invalid senders and threads, batch isolation, idempotent replies and publication, and commits/pushes against a local bare Git repository. Playwright tests desktop and mobile layouts using isolated mock artifacts and the empty production manifest.

Real validation has completed for Z-Image-Turbo generation on one Caviness A100 80GB, SSH result retrieval, private email preview generation, and GitHub Pages build/deployment. CI and mock tests do not substitute for real GPU validation. FLUX, A100 40GB, actual preemption/forced timeout, and Gmail OAuth sending still require validation. Generated candidates have not been approved and are absent from the public manifest.

## Project structure

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
│   ├── caviness-zimage-constraints.txt
│   ├── cron.example
│   ├── demo.sh
│   ├── demo.yaml
│   ├── e2e.sh
│   ├── hpc-python.sh
│   ├── hpc-runtime.env.example
│   └── slurm.sbatch
├── artcurator/
│   ├── __init__.py
│   ├── __main__.py
│   ├── settings.py
│   ├── store.py
│   ├── curator/{__init__,publish}.py
│   ├── generator/{__init__,backends,flux,plan,resources,worker}.py
│   ├── mailer/{__init__,format,replies,service,transport}.py
│   ├── preference_engine/{__init__,engine}.py
│   └── scheduler/{__init__,service,slurm}.py
├── tests/
│   ├── conftest.py
│   ├── test_generation_scheduler.py
│   ├── test_gpu_allocation.py
│   ├── test_hpc_container.py
│   ├── test_mail_publish.py
│   └── test_resources.py
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
