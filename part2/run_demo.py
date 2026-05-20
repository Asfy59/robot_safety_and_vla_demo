"""
Part 2 — VLA Demo: SmolVLA running a LIBERO manipulation task.

Architecture:
    Camera images + Robot state + Language instruction
              ↓
    SmolVLA (Vision-Language-Action model, 500M params)
              ↓
    Continuous 7-DoF end-effector actions (50-step chunks)
              ↓
    MuJoCo simulator (Franka Panda arm)

The language instruction is the core of the VLA: the model was trained on
human demonstrations paired with natural language task descriptions. At
inference time we pass a task string (e.g. "pick up the yellow book") and
the model generates motor commands conditioned on both vision and language.

Usage:
    # With a display (opens a live window):
    DISPLAY=:0 MUJOCO_GL=glfw python3 run_demo.py

    # Headless (saves demo.mp4):
    MUJOCO_GL=osmesa python3 run_demo.py --headless

    # Choose task 0-9:
    MUJOCO_GL=osmesa python3 run_demo.py --task_id 3 --headless
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--task_id",   type=int,  default=0)
parser.add_argument("--max_steps", type=int,  default=300)
parser.add_argument("--headless",  action="store_true")
parser.add_argument("--output",    type=str,  default="demo.mp4")
args = parser.parse_args()

# ── 1. Load the VLA policy ────────────────────────────────────────────────────
print("[demo] Loading SmolVLA (lerobot/smolvla_libero)...")
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

POLICY_ID = "lerobot/smolvla_libero"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = SmolVLAPolicy.from_pretrained(POLICY_ID).to(device).eval()
print(f"[demo] Policy ready on {device}")

# ── 2. Build the simulator ────────────────────────────────────────────────────
print(f"[demo] Building LIBERO env (task_id={args.task_id})...")
from lerobot.envs.libero import LiberoEnv, _get_suite

task_suite_name = "libero_object"
task_suite = _get_suite(task_suite_name)

env = LiberoEnv(
    task_suite=task_suite,
    task_id=args.task_id,
    task_suite_name=task_suite_name,
    render_mode="rgb_array",
    obs_type="pixels_agent_pos",
    observation_height=360,
    observation_width=360,
)

# ── 3. Get the language instruction for this task ─────────────────────────────
# This is the key VLA input: natural language describing what the robot must do.
# SmolVLA uses this alongside vision to decide which actions to take.
task_lang = getattr(env, "task_language", None) \
         or getattr(env, "task_name", None) \
         or task_suite.get_task(args.task_id).language
print(f"[demo] Language instruction: '{task_lang}'")

# ── video writer ──────────────────────────────────────────────────────────────
writer = None
if args.headless:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, 15, (720, 360))
    print(f"[demo] Saving to {args.output}")

# ── 4. Run the VLA→action loop ────────────────────────────────────────────────
from lerobot.envs.utils import preprocess_observation

# Camera name mapping: LiberoEnv image keys → SmolVLA expected keys
CAMERA_RENAME = {
    "observation.images.agentview_image":          "observation.images.camera1",
    "observation.images.robot0_eye_in_hand_image": "observation.images.camera2",
}

obs, _ = env.reset()
policy.reset()
success = False
print(f"[demo] Running (max {args.max_steps} steps)…")

for step in range(args.max_steps):

    # --- Build the VLA input batch -------------------------------------------
    # Step 1: convert numpy obs (pixels dict + agent_pos) to tensors
    batch = preprocess_observation(obs)

    # Step 2: rename camera keys to match SmolVLA expectations
    batch = {CAMERA_RENAME.get(k, k): v for k, v in batch.items()}

    # Step 3: move to GPU
    batch = {k: v.to(device) for k, v in batch.items()}

    # Step 4: add language instruction — the "VL" part of VLA
    batch["task"] = [task_lang]

    # --- VLA inference: vision + language → action ---------------------------
    with torch.inference_mode():
        action = policy.select_action(batch)
    action_np = action.squeeze(0).cpu().numpy()

    # --- Step the simulator --------------------------------------------------
    obs, reward, terminated, truncated, info = env.step(action_np)

    # --- Render ---------------------------------------------------------------
    frame = env.render()
    if frame is None:
        frame = np.zeros((360, 360, 3), dtype=np.uint8)

    cv2.putText(frame, f"step {step+1:03d}  reward={reward:.2f}",
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"task: {task_lang[:45]}",
                (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)
    cv2.putText(frame, f"action: [{', '.join(f'{a:.2f}' for a in action_np[:4])}...]",
                (8, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 255), 1)

    if args.headless:
        # wrist cam side-by-side
        wrist_raw = obs.get("observation.images.image2",
                            np.zeros((360, 360, 3), np.uint8))
        if isinstance(wrist_raw, np.ndarray) and wrist_raw.dtype != np.uint8:
            wrist_raw = (wrist_raw * 255).astype(np.uint8)
        side = np.concatenate([frame, wrist_raw], axis=1)
        writer.write(cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
    else:
        cv2.imshow("SmolVLA — LIBERO", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if terminated or truncated:
        success = info.get("success", False) or reward > 0
        print(f"[demo] Finished at step {step+1} | success={success}")
        break

print(f"[demo] success={success}")
if writer:
    writer.release()
    print(f"[demo] Saved → {args.output}")
else:
    cv2.destroyAllWindows()
env.close()
