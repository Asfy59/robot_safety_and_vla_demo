# Part II — VLA Model Integration

Runs a pre-trained **Vision-Language-Action (VLA)** model in a MuJoCo simulator, demonstrating end-to-end inference: camera images + robot state + language instruction → continuous motor actions.

---

## Stack

| Component | Choice | Why |
|---|---|---|
| **VLA model** | [SmolVLA](https://huggingface.co/lerobot/smolvla_libero) (`smolvla_libero`) | Compact 500M-param VLA fine-tuned on LIBERO; runs on a single GPU; open weights |
| **Simulator** | MuJoCo via [LIBERO](https://huggingface.co/lerobot/smolvla_libero) (`hf-libero`) | Standard manipulation benchmark; Franka Panda arm; 10 object-manipulation tasks |
| **Framework** | [LeRobot 0.4.4](https://github.com/huggingface/lerobot) (HuggingFace) | Unified policy + eval tooling; pre-built LIBERO integration |
| **Hardware** | RTX 5090, CUDA 12.8 | ~17 Hz inference |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  MuJoCo / LIBERO simulator                              │
│  Franka Panda arm — libero_object suite (10 tasks)      │
│                                                          │
│  obs = {                                                 │
│    pixels/agentview_image   (360×360×3)  ──┐            │
│    pixels/wrist_image       (360×360×3)  ──┤──► SmolVLA │
│    robot_state/eef/pos      (3,)         ──┤   (500M)   │
│    robot_state/joints/pos   (7,)         ──┘     │      │
│    language: "pick up the yellow book"  ─────────┘      │
│  }                                               │      │
│                                                  ▼      │
│  action = [Δx Δy Δz Δrx Δry Δrz gripper]×50 steps     │
│                    │                                    │
│                env.step(action)                         │
│                    │                                    │
│              next observation                           │
└──────────────────────────────────────────────────────────┘
```

The **language instruction** is what makes this a VLA (not just a visuomotor policy): the same weights handle all 10 tasks by conditioning on the task description string. SmolVLA uses a SmolVLM2-500M vision-language backbone fused with a flow-matching action head.

---

## Results

Evaluated on `libero_object` suite (10 tasks × 10 episodes = 100 total) using `lerobot-eval`:

| Metric | Value |
|---|---|
| **Overall success rate** | **71% (71/100)** |
| Best task (task 8) | 100% (10/10) |
| Avg episode length | ~12 s |
| Total eval time | ~20 min |

Per-task breakdown:

| Task | Language instruction | Success | Rate |
|---|---|---|---|
| 0 | pick up the **alphabet soup** and place it in the basket | 6/10 | 60% |
| 1 | pick up the **bbq sauce** and place it in the basket | 7/10 | 70% |
| 2 | pick up the **butter** and place it in the basket | 8/10 | 80% |
| 3 | pick up the **chocolate pudding** and place it in the basket | 6/10 | 60% |
| 4 | pick up the **cream cheese** and place it in the basket | 8/10 | 80% |
| 5 | pick up the **ketchup** and place it in the basket | 5/10 | 50% |
| 6 | pick up the **milk** and place it in the basket | 9/10 | 90% |
| 7 | pick up the **orange juice** and place it in the basket | 6/10 | 60% |
| 8 | pick up the **salad dressing** and place it in the basket | 10/10 | **100%** |
| 9 | pick up the **tomato sauce** and place it in the basket | 6/10 | 60% |

All tasks share the same scene setup and robot, but the **language instruction alone** tells the model which object to pick up — this is the core VLA capability.

Episode videos saved to `outputs/eval/2026-05-20/15-26-38_libero_smolvla/videos/`.

Hardware: RTX 5090, CUDA 12.8 — ~17 Hz inference, ~13 s/episode.

---

## Quickstart

```bash
# 1. Create isolated environment
python3.10 -m venv ~/venvs/asfand
source ~/venvs/asfand/bin/activate

# 2. Install dependencies
pip install lerobot
pip install "lerobot[libero]" --no-build-isolation
sudo apt install libosmesa6   # headless rendering

# 3. Run eval (headless)
MUJOCO_GL=osmesa lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --policy.empty_cameras=1

# 4. Run live demo (requires display)
DISPLAY=:0 MUJOCO_GL=glfw python3 run_demo.py --task_id 0
```

> **Note on `--rename_map`:** The LIBERO environment names its cameras `image` / `image2`; SmolVLA expects `camera1` / `camera2` / `camera3`. The rename_map bridges this, and `--policy.empty_cameras=1` pads the missing third camera with zeros.

---

## Why we show recordings, not a live MuJoCo window

The VLA **does run live inference** during eval (~17 Hz: obs → policy → action → sim step). What we replay are the **frames captured during that live loop**, not a separate offline animation.

We use saved `.mp4` episodes for the demo instead of a pop-up simulator window because:

| Issue | What happened |
|---|---|
| **Headless eval path** | `MUJOCO_GL=osmesa` renders off-screen to a buffer and writes video files. No window is created — by design. |
| **EGL parallel contexts** | `MUJOCO_GL=egl` with `batch_size>1` crashed robosuite's EGL context cleanup before any episode finished. |
| **GLFW + `lerobot-eval`** | Even with `DISPLAY=:0` and `MUJOCO_GL=glfw`, LeRobot's eval loop uses `render_mode=rgb_array` and saves videos; it does not open an interactive MuJoCo viewer window. |
| **Custom live script** | `run_demo.py` would need the same obs/preprocessor pipeline as `lerobot-eval` — still WIP. |

**Bottom line:** the policy integration is proven by the 71% success rate and 100 saved episode videos. The Flask viewer replays those recordings with task labels and success/failure markers — same physics, same policy, just not rendered in real time on screen.

A true interactive demo (pick task → type instruction → watch arm move in a live window) is the next step; it needs either a GLFW render loop in a custom script or streaming osmesa frames to the browser.

---

## Episode viewer (recommended demo)

Browse all 100 episodes by task, with success/failure labels:

```bash
source ~/venvs/asfand/bin/activate
pip install flask
cd ~/asfand_challenge
python3 part2/viewer.py
# Open http://localhost:5050
```

Click a task → click an episode → video plays in the browser with the language instruction and result shown.

---

## Watch saved episodes (VLC)

The eval saves one `.mp4` per episode. Play back the best results:

```bash
# Task 8 — 100% success rate (best task)
vlc ~/asfand_challenge/outputs/eval/2026-05-20/15-26-38_libero_smolvla/videos/libero_object_8/eval_episode_0.mp4

# Task 6 — 90% success rate
vlc ~/asfand_challenge/outputs/eval/2026-05-20/15-26-38_libero_smolvla/videos/libero_object_6/eval_episode_0.mp4

# Or play all 100 episodes sequentially
vlc ~/asfand_challenge/outputs/eval/2026-05-20/15-26-38_libero_smolvla/videos/libero_object_*/eval_episode_0.mp4
```

## Re-run eval (generates new recordings)

To run the policy again and save fresh episode videos:

```bash
source ~/venvs/asfand/bin/activate
cd ~/asfand_challenge

MUJOCO_GL=osmesa lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --policy.empty_cameras=1
```

## File layout

```
part2/
├── README.md       this file
├── viewer.py       Flask episode browser (task picker + video replay)
└── run_demo.py     standalone live demo script (work in progress)
```

---

## Key learnings from this challenge

### What a VLA actually does
A VLA is not magic — it's a transformer that takes (image, robot state, language string) as tokens and outputs a sequence of joint/end-effector actions. The "vision-language" part is a pre-trained image+text encoder (SmolVLM2 here); the "action" part is a flow-matching head on top. The language string is just another input token — the same weights serve all 10 tasks.

### Fine-tuning vs base model matters enormously
`smolvla_base` scored 0%. `smolvla_libero` (same architecture, fine-tuned on LIBERO demonstrations) scored 71%. Pre-trained VLAs are not zero-shot manipulation models — they need task-domain fine-tuning to be useful. The base model has general visual reasoning but has never seen a robot arm or a basket.

### Simulator integration is mostly a naming problem
The hardest part of the integration was not the model inference — it was bridging the observation key names between the LIBERO environment (`image`, `image2`) and the SmolVLA checkpoint (`camera1`, `camera2`, `camera3`). The `--rename_map` argument solves this pattern. Any real deployment will have a similar bridging step between the robot's sensor API and the policy's expected input format.

### Rendering backends matter for headless setups
MuJoCo has three backends: GLFW (needs a display), EGL (headless GPU, complex context management), OSMesa (headless CPU, always works). For parallel environments, EGL context management in robosuite breaks above ~10 parallel contexts. OSMesa + `sudo apt install libosmesa6` is the reliable headless path.

### Version pinning is critical for VLA models
`pi0_libero_finetuned` (unpinned) failed with a transformers version check. `pi0_libero_finetuned_v044` (pinned to lerobot 0.4.4) would have worked. Always check for version-specific model checkpoints before debugging dependency errors.

---

## Known limitations & next steps

- **Fine-tuning:** `smolvla_libero` was fine-tuned on LIBERO object tasks only. Other LIBERO suites (spatial, goal, long-horizon) would need additional fine-tuning.
- **Rendering:** EGL headless rendering fails with >10 parallel robosuite contexts. Use `MUJOCO_GL=osmesa` (CPU software rendering) or `MUJOCO_GL=glfw` with a display.
- **Robot model:** The Agibot G2 URDF is available at `robot_description/` but visual meshes are not bundled and no pre-trained VLA checkpoint exists for G2. Integration would require mesh sourcing + URDF→MJCF conversion + task-specific fine-tuning.
- **ROS2 bridge:** Actions could be published to `/vla/action` and gated by the Part I safety backend (`velocity_factor`) for a full end-to-end safety-aware manipulation loop.
