"""Stream pinned HF weights through the private device to HPC without local cache.

No GPU, no remote HF token. Each file is verified against Hub metadata before
atomic rename; interrupted .part files resume via HTTP Range on the next run.
Run with HF_HOME pointing to the private device's HF credential directory.
"""
import argparse
from fnmatch import fnmatch
import importlib.util
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess


def safe_relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\n" in name or "\r" in name:
        raise ValueError("Unsafe model file path")
    return path


def verification(path, size, digest, lfs):
    if not re.fullmatch(r"[a-f0-9]{64}" if lfs else r"[a-f0-9]{40}", digest):
        raise ValueError("Missing pinned file checksum")
    path = shlex.quote(str(path))
    hasher = f"sha256sum {path}" if lfs else f"{{ printf 'blob {int(size)}\\0'; cat {path}; }} | sha1sum"
    return (f"[ -f {path} ] && [ \"$(wc -c < {path})\" -eq {int(size)} ] && "
            f"[ \"$({hasher} | cut -d ' ' -f 1)\" = {shlex.quote(digest)} ]")


def ssh(host, script, **kwargs):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                           "-o", "StrictHostKeyChecking=yes", host, script], **kwargs)


def transfer(host, destination, url, token, size, digest, lfs):
    import httpx
    target = shlex.quote(str(destination))
    part = shlex.quote(str(destination) + ".part")
    parent = shlex.quote(str(destination.parent))
    if ssh(host, verification(destination, size, digest, lfs), capture_output=True).returncode == 0:
        return "verified cached"
    probe = ssh(host, f"umask 077; mkdir -p {parent} && if [ -f {part} ]; then wc -c < {part}; else echo 0; fi",
                capture_output=True, text=True, check=True)
    offset = int(probe.stdout.strip())
    if offset >= size:
        offset = 0
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    # httpx strips Authorization on cross-origin redirects; the HF token stays
    # in this private process, never in SSH argv, remote files or model paths.
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            if offset and response.status_code == 206:
                if not response.headers.get("content-range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Incorrect download resume range")
            else:
                offset = 0
            redirect = ">>" if offset else ">"
            verify = verification(str(destination) + ".part", size, digest, lfs)
            command = f"umask 077; cat {redirect} {part} && {verify} && mv {part} {target}"
            with subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                   "-o", "StrictHostKeyChecking=yes", host, command],
                                  stdin=subprocess.PIPE, stdout=subprocess.DEVNULL) as process:
                try:
                    for chunk in response.iter_bytes(1024 * 1024):
                        process.stdin.write(chunk)
                    process.stdin.close()
                    if process.wait() != 0:
                        raise RuntimeError("Transfer/checksum failed; partial file retained for retry")
                finally:
                    if not process.stdin.closed:
                        process.stdin.close()
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
    return "transferred and verified"


def main():
    from huggingface_hub import HfApi, get_token, hf_hub_url
    spec = importlib.util.spec_from_file_location("downloads", Path(__file__).with_name("download-flux.py"))
    downloads = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(downloads)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-root", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", args.host):
        raise ValueError("Invalid SSH host")
    root = PurePosixPath(args.remote_root)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("Remote cache root must be absolute")
    api = HfApi()
    token = get_token()
    for repo, revision, folder, patterns in downloads.MODELS:
        api.auth_check(repo_id=repo, repo_type="model")
    for repo, revision, folder, patterns in downloads.MODELS:
        info = api.model_info(repo, revision=revision, files_metadata=True)
        for file in info.siblings:
            if not any(fnmatch(file.rfilename, pattern) for pattern in patterns):
                continue
            name = safe_relative(file.rfilename)
            if file.size is None:
                raise ValueError("Missing pinned file size")
            digest = file.lfs.sha256 if file.lfs else file.blob_id
            result = transfer(args.host, root / folder / name, hf_hub_url(repo, str(name), revision=revision),
                              token, file.size, digest, bool(file.lfs))
            print(f"{repo}/{name}: {result}", flush=True)


if __name__ == "__main__":
    main()
