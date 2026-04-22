"""
Tier 1 Colab runner: resume + deterministic L-BFGS tail + Drive sync.

Best-practice profile for the patched Tier 1 trainer:
- Adam phase ends at epoch 120
- Single L-BFGS epoch (epoch 121) with max_iter=100
- Fixed closure subset via TIER1_LBFGS_MAX_BATCHES (trainer caches once)
"""

import os
import re
import sys
import time
import shutil
import subprocess
from pathlib import Path

import torch
from google.colab import drive, userdata


# -------------------------
# CONFIG
# -------------------------
GITHUB_TOKEN = userdata.get("GITHUB_TOKEN")
REPO_OWNER = "sylpagnier"
REPO_NAME = "LadHyX_ml_cfd_thrombus_predictions"
BRANCH_NAME = "master"

ZIP_PATH = Path("/content/drive/MyDrive/AI_Thrombosis_Code_Zipped_Folders/graphs_tier1.zip")
CHECKPOINT_PATH = Path("/content/drive/MyDrive/AI_Thrombosis_Code_Zipped_Folders/tier1_latest_checkpoint.pth")

# Main backup folder
BACKUP_DIR = Path("/content/drive/MyDrive/AI_Thrombosis_Code_Zipped_Folders/LBFGS_Results_T1_BEST_PRACTICE")
# Durable "latest-resume" folder (always overwritten with newest ckpt)
RESUME_SYNC_DIR = Path("/content/drive/MyDrive/AI_Thrombosis_Code_Zipped_Folders/Tier1_Resume_Sync")

# Best-practice schedule:
# Adam epochs [0..119], one L-BFGS epoch at 120.
TIER1_ADAM_EPOCHS = "120"
TIER1_EPOCHS = "121"

# Memory/runtime controls
MAX_LOAD = "120"                  # try 80 if still memory-constrained
LBFGS_MAX_ITER = "100"            # single continuous L-BFGS solve
LBFGS_HISTORY_SIZE = "10"
LBFGS_LR = "0.01"
LBFGS_MAX_BATCHES = "12"          # static closure subset size

# Sync controls
SYNC_EVERY_EPOCH = True
SYNC_EVERY_SEC = 120

if not GITHUB_TOKEN:
    raise RuntimeError("Missing GITHUB_TOKEN in Colab secrets.")
REPO_URL = f"https://{GITHUB_TOKEN}@github.com/{REPO_OWNER}/{REPO_NAME}.git"


def run(cmd, check=True):
    print(">", " ".join(cmd))
    return subprocess.run(cmd, check=check)


def pip(*args):
    run(["pip", *args])


def safe_copy(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def sync_checkpoints_to_drive(repo_root: Path):
    """
    Copies latest/best checkpoints + diary into durable Drive paths.
    Safe to call repeatedly.
    """
    stage_a = repo_root / "outputs" / "stage_a"
    reports = repo_root / "outputs" / "reports"

    latest_ckpt = stage_a / "tier1_latest_checkpoint.pth"
    best_loss = stage_a / "tier1_best_loss.pth"
    best_phys = stage_a / "tier1_best_physics.pth"

    # 1) Stable resume location (always overwrite)
    safe_copy(latest_ckpt, RESUME_SYNC_DIR / "tier1_latest_checkpoint.pth")
    safe_copy(best_loss, RESUME_SYNC_DIR / "tier1_best_loss.pth")
    safe_copy(best_phys, RESUME_SYNC_DIR / "tier1_best_physics.pth")

    # 2) Session backup location
    stage_a_backup = BACKUP_DIR / "stage_a"
    safe_copy(latest_ckpt, stage_a_backup / "tier1_latest_checkpoint.pth")
    safe_copy(best_loss, stage_a_backup / "tier1_best_loss.pth")
    safe_copy(best_phys, stage_a_backup / "tier1_best_physics.pth")

    # copy newest diary if present
    if reports.exists():
        diaries = sorted(reports.glob("training_diary_tier1_*.jsonl"))
        if diaries:
            safe_copy(diaries[-1], BACKUP_DIR / "reports" / diaries[-1].name)


print("📂 Mounting Google Drive...")
drive.mount("/content/drive", force_remount=True)
os.chdir("/content")

BACKUP_DIR.mkdir(parents=True, exist_ok=True)
RESUME_SYNC_DIR.mkdir(parents=True, exist_ok=True)

print(f"🔎 Current torch: {torch.__version__}")
if "2.5.1+cu121" not in torch.__version__:
    print("⬇️ Installing torch 2.5.1+cu121 stack...")
    pip("uninstall", "-y", "torch", "torchvision", "torchaudio")
    pip(
        "install",
        "torch==2.5.1+cu121",
        "torchvision==0.20.1+cu121",
        "torchaudio==2.5.1+cu121",
        "--index-url",
        "https://download.pytorch.org/whl/cu121",
    )

print("⬇️ Installing PyG wheels...")
wheel_url = "https://data.pyg.org/whl/torch-2.5.1+cu121.html"
pip("install", "-q", "pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv", "-f", wheel_url)
pip("install", "-q", "torch-geometric", "matplotlib", "tqdm")

repo_path = Path("/content") / REPO_NAME
if repo_path.exists():
    print("🔄 Removing old repo copy...")
    shutil.rmtree(repo_path)

print(f"⬇️ Cloning {REPO_NAME} ({BRANCH_NAME})...")
run(["git", "clone", "-b", BRANCH_NAME, REPO_URL])
os.chdir(repo_path)

target_data_dir = Path("data/processed")
target_data_dir.mkdir(parents=True, exist_ok=True)

if not ZIP_PATH.exists():
    raise FileNotFoundError(f"Data zip not found: {ZIP_PATH}")

print(f"📦 Extracting {ZIP_PATH} -> {target_data_dir}")
run(["unzip", "-q", "-o", str(ZIP_PATH), "-d", str(target_data_dir)])

expected_graph_dir = target_data_dir / "graphs_tier1"
pt_files = sorted(expected_graph_dir.glob("vessel_*.pt"))
print(f"🔍 Found {len(pt_files)} graph files in {expected_graph_dir}")
if not pt_files:
    raise RuntimeError(f"No vessel_*.pt found under {expected_graph_dir}")

ckpt_dir = Path("outputs/stage_a")
ckpt_dir.mkdir(parents=True, exist_ok=True)

resume_synced_ckpt = RESUME_SYNC_DIR / "tier1_latest_checkpoint.pth"
source_ckpt = resume_synced_ckpt if resume_synced_ckpt.exists() else CHECKPOINT_PATH
if not source_ckpt.exists():
    raise FileNotFoundError(
        f"No checkpoint found. Checked:\n- {resume_synced_ckpt}\n- {CHECKPOINT_PATH}"
    )

dst_ckpt = ckpt_dir / "tier1_latest_checkpoint.pth"
shutil.copy(source_ckpt, dst_ckpt)
print(f"📥 Copied checkpoint -> {dst_ckpt} (source: {source_ckpt})")

os.environ["PYTHONPATH"] = str(repo_path)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONUNBUFFERED"] = "1"

os.environ["TIER1_RESUME"] = "1"
os.environ["TIER1_USE_LBFGS"] = "1"
os.environ["TIER1_STOP_AFTER_ADAM"] = "0"
os.environ["TIER1_ADAM_EPOCHS"] = TIER1_ADAM_EPOCHS
os.environ["TIER1_EPOCHS"] = TIER1_EPOCHS

os.environ["TIER1_MAX_LOAD_VESSELS"] = MAX_LOAD
os.environ["TIER1_MAX_LOAD_SHUFFLE"] = "1"

os.environ["TIER1_LBFGS_MAX_ITER"] = LBFGS_MAX_ITER
os.environ["TIER1_LBFGS_HISTORY_SIZE"] = LBFGS_HISTORY_SIZE
os.environ["TIER1_LBFGS_LR"] = LBFGS_LR
os.environ["TIER1_LBFGS_MAX_BATCHES"] = LBFGS_MAX_BATCHES

os.environ["TIER1_SKIP_VALIDATION"] = "1"
os.environ["TIER1_DISABLE_FIGURES"] = "1"
os.environ["TIER1_MICRO_BATCH_SIZE"] = "1"
os.environ["TIER1_ACCUMULATION_STEPS"] = "8"
os.environ["TIER1_CKPT_EVERY"] = "1"
os.environ["TIER1_EXPERIMENT_NAME"] = "tier1_colab_lbfgs_best_practice_sync"

print("\n🧪 Diagnostics")
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
run(["nvidia-smi"], check=False)

sync_checkpoints_to_drive(repo_path)
print(f"✅ Initial sync complete -> {RESUME_SYNC_DIR}")

print("\n🚀 Resuming Tier 1 (single-epoch L-BFGS phase, periodic sync)...\n")
cmd = [sys.executable, "-u", "-m", "src.training.train_t1_predictor", "--resume"]
print("Command:", " ".join(cmd))

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env=os.environ.copy(),
)

start_t = time.time()
last_output_t = time.time()
last_sync_t = 0.0

max_silence_sec = 3600
max_runtime_sec = 10 * 3600

epoch_done_re = re.compile(r"Saved Tier 1 checkpoint -> tier1_latest_checkpoint\.pth")

try:
    while True:
        line = proc.stdout.readline()
        if line:
            print(line, end="")
            last_output_t = time.time()

            if SYNC_EVERY_EPOCH and epoch_done_re.search(line):
                sync_checkpoints_to_drive(repo_path)
                print("🔄 Synced checkpoints to Drive (epoch checkpoint detected).")

            if time.time() - last_sync_t >= SYNC_EVERY_SEC:
                sync_checkpoints_to_drive(repo_path)
                last_sync_t = time.time()

        elif proc.poll() is not None:
            break
        else:
            now = time.time()
            if now - last_output_t > max_silence_sec:
                raise TimeoutError(f"No output for {max_silence_sec}s.")
            if now - start_t > max_runtime_sec:
                raise TimeoutError(f"Exceeded runtime guard ({max_runtime_sec}s).")
            if now - last_sync_t >= SYNC_EVERY_SEC:
                sync_checkpoints_to_drive(repo_path)
                last_sync_t = now
            time.sleep(1.0)

except Exception as e:
    print(f"\n[watchdog] {e}")
    print("[watchdog] Terminating process...")
    proc.terminate()
    time.sleep(5)
    if proc.poll() is None:
        proc.kill()
    raise

finally:
    rc = proc.wait()
    print(f"\nProcess exit code: {rc}")

    sync_checkpoints_to_drive(repo_path)
    print(f"🔒 Final sync complete -> {RESUME_SYNC_DIR}")

    if rc != 0:
        raise RuntimeError(f"Training failed with exit code {rc}")

print("\n💾 Final backup snapshot...")
(BACKUP_DIR / "stage_a").mkdir(parents=True, exist_ok=True)
(BACKUP_DIR / "reports").mkdir(parents=True, exist_ok=True)
run(["bash", "-lc", f"rsync -a --delete outputs/stage_a/ '{BACKUP_DIR / 'stage_a'}/'"], check=False)
run(["bash", "-lc", f"rsync -a outputs/reports/ '{BACKUP_DIR / 'reports'}/'"], check=False)

print(f"✅ Done. Resume-safe checkpoint path: {RESUME_SYNC_DIR / 'tier1_latest_checkpoint.pth'}")
print(f"✅ Session backup path: {BACKUP_DIR}")
