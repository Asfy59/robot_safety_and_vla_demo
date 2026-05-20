"""
Part 2 — Episode Viewer
Flask server to browse and replay SmolVLA-LIBERO evaluation episodes.

Usage:
    python3 viewer.py
    Open http://localhost:5050 in a browser.
"""

import os
from pathlib import Path
from flask import Flask, render_template_string, send_file, abort

app = Flask(__name__)

EVAL_DIR = Path(__file__).parent.parent / "outputs/eval/2026-05-20/15-26-38_libero_smolvla/videos"

TASKS = {
    0: {"name": "alphabet soup",      "rate": 60,  "successes": [1,0,0,1,1,1,0,1,0,1]},
    1: {"name": "bbq sauce",          "rate": 70,  "successes": [1,1,1,0,1,1,0,1,1,0]},
    2: {"name": "butter",             "rate": 80,  "successes": [1,0,1,1,1,1,1,1,0,1]},
    3: {"name": "chocolate pudding",  "rate": 60,  "successes": [1,1,0,1,1,1,0,1,0,0]},
    4: {"name": "cream cheese",       "rate": 80,  "successes": [1,1,0,1,1,1,0,1,1,1]},
    5: {"name": "ketchup",            "rate": 50,  "successes": [1,0,1,0,0,0,1,1,0,1]},
    6: {"name": "milk",               "rate": 90,  "successes": [1,1,1,1,1,1,1,1,1,0]},
    7: {"name": "orange juice",       "rate": 60,  "successes": [1,0,1,1,0,1,0,0,1,1]},
    8: {"name": "salad dressing",     "rate": 100, "successes": [1,1,1,1,1,1,1,1,1,1]},
    9: {"name": "tomato sauce",       "rate": 60,  "successes": [0,1,0,1,1,1,0,0,1,1]},
}

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>SmolVLA — LIBERO Episode Viewer</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #0f1117; color: #e0e0e0; }
    header { padding: 24px 32px; border-bottom: 1px solid #2a2a3a; }
    header h1 { font-size: 1.4rem; color: #fff; }
    header p  { font-size: 0.85rem; color: #888; margin-top: 4px; }
    .badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; }
    .badge.success { background:#1a3a1a; color:#4caf50; }
    .badge.model   { background:#1a2a3a; color:#64b5f6; }
    .layout { display: flex; height: calc(100vh - 73px); }
    .sidebar { width: 340px; min-width:340px; overflow-y: auto; border-right: 1px solid #2a2a3a; padding: 16px; }
    .sidebar h2 { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #666; margin-bottom: 12px; }
    .task-card { border: 1px solid #2a2a3a; border-radius: 8px; margin-bottom: 8px; overflow: hidden; }
    .task-header { padding: 10px 14px; cursor: pointer; display:flex; align-items:center; gap:10px; }
    .task-header:hover { background: #1a1a2a; }
    .task-header.active { background: #1a1a3a; border-color: #3a3a6a; }
    .task-num { font-size:0.7rem; color:#666; min-width:18px; }
    .task-label { flex:1; font-size:0.85rem; }
    .task-label span { font-weight:600; color:#fff; }
    .rate-bar { height:4px; background:#2a2a3a; border-radius:2px; margin-top:4px; }
    .rate-fill { height:4px; border-radius:2px; }
    .episodes { padding: 8px 14px 12px; display:none; }
    .episodes.open { display:block; }
    .ep-row { display:flex; gap:6px; flex-wrap:wrap; margin-top:6px; }
    .ep-btn { padding:4px 10px; border-radius:4px; border:1px solid #2a2a3a; background:#0f1117;
              color:#aaa; font-size:0.75rem; cursor:pointer; text-decoration:none; }
    .ep-btn:hover { border-color:#5a5aaa; color:#fff; }
    .ep-btn.success { border-color:#2a4a2a; color:#4caf50; }
    .ep-btn.fail    { border-color:#4a2a2a; color:#ef5350; }
    .ep-btn.playing { background:#2a2a5a; border-color:#5a5aff; color:#fff; }
    .main { flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:32px; }
    .player-wrap { width:100%; max-width:700px; }
    .player-wrap video { width:100%; border-radius:8px; background:#000; }
    .player-info { margin-top:14px; }
    .player-info h2 { font-size:1.1rem; color:#fff; }
    .player-info p  { font-size:0.82rem; color:#888; margin-top:4px; }
    .placeholder { text-align:center; color:#444; }
    .placeholder svg { width:64px; opacity:0.3; margin-bottom:16px; }
    .stat-bar { display:flex; gap:24px; margin-top:16px; }
    .stat { text-align:center; }
    .stat .val { font-size:1.6rem; font-weight:700; color:#fff; }
    .stat .lbl { font-size:0.72rem; color:#666; text-transform:uppercase; letter-spacing:.08em; }
  </style>
</head>
<body>
<header>
  <h1>SmolVLA — LIBERO Object Manipulation</h1>
  <p>
    <span class="badge model">lerobot/smolvla_libero</span>&nbsp;
    <span class="badge success">71% overall success — 100 episodes</span>
    &nbsp;·&nbsp; Franka Panda · MuJoCo · RTX 5090
  </p>
</header>
<div class="layout">
  <div class="sidebar">
    <h2>10 Tasks · 10 episodes each</h2>
    {% for tid, task in tasks.items() %}
    {% set color = '#4caf50' if task.rate >= 80 else '#ff9800' if task.rate >= 60 else '#ef5350' %}
    <div class="task-card" id="card-{{tid}}">
      <div class="task-header" onclick="toggleTask({{tid}})">
        <span class="task-num">{{tid}}</span>
        <div style="flex:1">
          <div class="task-label">pick up the <span>{{task.name}}</span></div>
          <div class="rate-bar"><div class="rate-fill" style="width:{{task.rate}}%;background:{{color}}"></div></div>
        </div>
        <span style="font-size:0.8rem;font-weight:700;color:{{color}}">{{task.rate}}%</span>
      </div>
      <div class="episodes" id="eps-{{tid}}">
        <div style="font-size:0.72rem;color:#666;margin-bottom:4px">Episodes (green=success, red=fail):</div>
        <div class="ep-row">
          {% for ep in range(10) %}
          {% set ok = task.successes[ep] %}
          <a class="ep-btn {{ 'success' if ok else 'fail' }}"
             href="#" onclick="playEp({{tid}}, {{ep}}, this); return false;">
            ep {{ep}} {{ '✓' if ok else '✗' }}
          </a>
          {% endfor %}
        </div>
      </div>
    </div>
    {% endfor %}
  </div>
  <div class="main" id="main">
    <div class="placeholder">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <circle cx="12" cy="12" r="10"/><polygon points="10,8 16,12 10,16"/>
      </svg>
      <div style="font-size:1rem;color:#555">Select a task and episode</div>
      <div style="font-size:0.8rem;color:#444;margin-top:6px">to replay a recorded episode</div>
      <div class="stat-bar" style="margin-top:32px;justify-content:center">
        <div class="stat"><div class="val">71%</div><div class="lbl">Success rate</div></div>
        <div class="stat"><div class="val">100</div><div class="lbl">Episodes</div></div>
        <div class="stat"><div class="val">10</div><div class="lbl">Tasks</div></div>
        <div class="stat"><div class="val">~12s</div><div class="lbl">Avg episode</div></div>
      </div>
    </div>
  </div>
</div>
<script>
  function toggleTask(tid) {
    const eps = document.getElementById('eps-' + tid);
    const hdr = document.getElementById('card-' + tid).querySelector('.task-header');
    const open = eps.classList.toggle('open');
    hdr.classList.toggle('active', open);
  }

  function playEp(tid, ep, btn) {
    document.querySelectorAll('.ep-btn.playing').forEach(b => b.classList.remove('playing'));
    btn.classList.add('playing');

    const tasks = {{ tasks_json|safe }};
    const task = tasks[tid];
    const url = `/video/${tid}/${ep}`;

    document.getElementById('main').innerHTML = `
      <div class="player-wrap">
        <video id="vid" src="${url}" controls autoplay loop></video>
        <div class="player-info">
          <h2>Task ${tid} — pick up the <b>${task.name}</b> and place it in the basket</h2>
          <p>Episode ${ep} &nbsp;·&nbsp;
            ${task.successes[ep] ?
              '<span style="color:#4caf50">✓ SUCCESS</span>' :
              '<span style="color:#ef5350">✗ FAILURE</span>'}
            &nbsp;·&nbsp; Task success rate: ${task.rate}%
          </p>
        </div>
      </div>`;
  }
</script>
</body>
</html>
"""

@app.route("/")
def index():
    import json
    tasks_json = {str(k): {"name": v["name"], "rate": v["rate"], "successes": v["successes"]}
                  for k, v in TASKS.items()}
    return render_template_string(HTML, tasks=TASKS, tasks_json=json.dumps(tasks_json))

@app.route("/video/<int:task_id>/<int:episode>")
def video(task_id, episode):
    path = EVAL_DIR / f"libero_object_{task_id}" / f"eval_episode_{episode}.mp4"
    if not path.exists():
        abort(404, f"Video not found: {path}")
    return send_file(str(path), mimetype="video/mp4")

if __name__ == "__main__":
    if not EVAL_DIR.exists():
        print(f"[viewer] WARNING: eval dir not found at {EVAL_DIR}")
        print("[viewer] Run the eval first to generate videos.")
    print("[viewer] Open http://localhost:5050 in your browser")
    app.run(host="0.0.0.0", port=5050, debug=False)
