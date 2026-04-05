"""
report/generate_report.py

Generates a self-contained HTML report from H-MAD DST experiment results.
The report includes:
  - Per-run metric cards (AHAR, TTD, DST Score)
  - Colour-coded transcript viewer with contradiction highlights
  - Cross-configuration comparison table
  - Chart.js bar and radar charts
"""

import json
import os
from datetime import datetime
from pathlib import Path

REPORT_OUTPUT = Path(__file__).parent / "report.html"


def generate_html_report(all_results: list[dict], output_path: str = None) -> str:
    """
    Generate a full HTML report.

    Args:
        all_results:  List of result dicts from the orchestrator.
        output_path:  Where to write the HTML file. Defaults to report/report.html.

    Returns:
        Path to the generated HTML file.
    """
    output_path = Path(output_path) if output_path else REPORT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = _build_html(all_results)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return str(output_path)


# ─── HTML Builder ─────────────────────────────────────────────────────────────

def _build_html(results: list[dict]) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    metric_cards = "\n".join(_render_metric_card(r) for r in results)
    transcript_sections = "\n".join(_render_transcript_section(r) for r in results)
    comparison_table = _render_comparison_table(results)
    chart_data_json = json.dumps(_build_chart_data(results), indent=2)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>H-MAD DST Experiment Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:        #0e0f14;
    --surface:   #171922;
    --surface2:  #1e2130;
    --border:    #2a2f45;
    --text:      #d4d8f0;
    --muted:     #7a82a8;
    --accent:    #5b7cf7;
    --green:     #3ecf8e;
    --red:       #e05c6c;
    --amber:     #f4a228;
    --teal:      #3acfcf;
    --radius:    10px;
    --font:      'IBM Plex Mono', 'Fira Code', monospace;
    --font-sans: 'IBM Plex Sans', system-ui, sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-sans);
    font-size: 15px;
    line-height: 1.65;
    padding: 0 0 80px;
  }}

  /* ── Header ── */
  .header {{
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 32px 48px 28px;
    position: sticky; top: 0; z-index: 100;
  }}
  .header h1 {{
    font-family: var(--font);
    font-size: 18px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.04em;
  }}
  .header .sub {{
    font-size: 13px;
    color: var(--muted);
    margin-top: 4px;
    font-family: var(--font);
  }}
  .nav-tabs {{
    display: flex; gap: 4px; margin-top: 20px;
  }}
  .nav-tab {{
    padding: 7px 18px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--muted);
    font-size: 13px;
    cursor: pointer;
    font-family: var(--font);
    transition: all 0.15s;
  }}
  .nav-tab:hover {{ color: var(--text); border-color: var(--accent); }}
  .nav-tab.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}

  /* ── Layout ── */
  .container {{ max-width: 1200px; margin: 0 auto; padding: 0 48px; }}
  .section {{ padding-top: 48px; }}
  .section-title {{
    font-family: var(--font);
    font-size: 13px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
  }}

  /* ── Metric Cards ── */
  .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
  }}
  .card-header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px; }}
  .card-title {{ font-size: 13px; color: var(--muted); font-family: var(--font); }}
  .config-badge {{
    font-size: 11px;
    font-family: var(--font);
    padding: 3px 10px;
    border-radius: 4px;
    background: var(--accent);
    color: #fff;
    font-weight: 600;
  }}
  .config-badge.b {{ background: var(--teal); color: var(--bg); }}

  .metric-row {{ display: flex; justify-content: space-between; margin-bottom: 10px; align-items: center; }}
  .metric-label {{ font-size: 13px; color: var(--muted); }}
  .metric-value {{
    font-family: var(--font);
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
  }}
  .metric-value.good {{ color: var(--green); }}
  .metric-value.bad  {{ color: var(--red); }}
  .metric-value.warn {{ color: var(--amber); }}
  .metric-value.na   {{ color: var(--muted); font-weight: 400; }}

  .score-bar-bg {{
    height: 6px; background: var(--surface2);
    border-radius: 3px; margin-top: 6px; overflow: hidden;
  }}
  .score-bar {{ height: 100%; border-radius: 3px; transition: width 0.6s ease; }}

  .model-row {{
    display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px;
    padding-top: 12px; border-top: 1px solid var(--border);
  }}
  .model-tag {{
    font-size: 11px;
    font-family: var(--font);
    padding: 3px 8px;
    border-radius: 4px;
    border: 1px solid var(--border);
    color: var(--muted);
  }}
  .model-tag span {{ font-weight: 600; color: var(--text); }}

  /* ── DST Score Big ── */
  .dst-big {{
    font-family: var(--font);
    font-size: 48px;
    font-weight: 700;
    text-align: center;
    margin: 8px 0 4px;
    line-height: 1;
  }}
  .dst-label {{ text-align: center; font-size: 12px; color: var(--muted); margin-bottom: 16px; }}

  /* ── Judge reasoning ── */
  .reasoning {{
    background: var(--surface2);
    border-left: 3px solid var(--accent);
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: var(--muted);
    line-height: 1.7;
    margin-top: 16px;
  }}

  /* ── Charts ── */
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .chart-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
  }}
  .chart-title {{ font-size: 13px; color: var(--muted); font-family: var(--font); margin-bottom: 16px; }}
  canvas {{ max-height: 280px; }}

  /* ── Transcript ── */
  .transcript-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 24px;
    overflow: hidden;
  }}
  .transcript-header {{
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
    background: var(--surface2);
  }}
  .transcript-header .title {{ font-family: var(--font); font-size: 13px; color: var(--text); }}
  .transcript-toggle {{
    font-size: 12px; color: var(--accent); cursor: pointer;
    font-family: var(--font); border: none; background: none;
    text-decoration: underline; text-underline-offset: 3px;
  }}
  .transcript-body {{ padding: 0; }}
  .msg {{
    padding: 14px 24px;
    border-bottom: 1px solid var(--border);
    display: grid;
    grid-template-columns: 80px 60px 1fr;
    gap: 16px;
    align-items: start;
  }}
  .msg:last-child {{ border-bottom: none; }}
  .msg.setup {{ opacity: 0.6; }}
  .msg.contradiction {{
    background: rgba(224, 92, 108, 0.08);
    border-left: 3px solid var(--red);
  }}
  .msg.detected {{
    background: rgba(62, 207, 142, 0.07);
    border-left: 3px solid var(--green);
  }}
  .msg-turn {{
    font-family: var(--font);
    font-size: 11px;
    color: var(--muted);
    padding-top: 2px;
  }}
  .msg-role {{
    font-family: var(--font);
    font-size: 11px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 4px;
    text-align: center;
    width: fit-content;
  }}
  .msg-role.user      {{ background: rgba(224, 92, 108, 0.15); color: var(--red); }}
  .msg-role.assistant {{ background: rgba(91, 124, 247, 0.15); color: var(--accent); }}
  .msg-role.setup     {{ background: var(--surface2); color: var(--muted); }}
  .msg-content {{ font-size: 13px; line-height: 1.7; }}
  .contradiction-badge {{
    display: inline-block; font-size: 10px;
    font-family: var(--font);
    background: rgba(224, 92, 108, 0.2);
    color: var(--red);
    border: 1px solid rgba(224, 92, 108, 0.4);
    padding: 1px 6px; border-radius: 3px;
    margin-bottom: 4px;
  }}
  .detection-badge {{
    display: inline-block; font-size: 10px;
    font-family: var(--font);
    background: rgba(62, 207, 142, 0.15);
    color: var(--green);
    border: 1px solid rgba(62, 207, 142, 0.3);
    padding: 1px 6px; border-radius: 3px;
    margin-bottom: 4px;
  }}

  /* ── Comparison Table ── */
  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{
    font-family: var(--font);
    font-size: 11px; font-weight: 600;
    color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 10px 16px; text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  td {{
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
    font-family: var(--font);
    font-size: 13px;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: var(--surface2); }}
  .better {{ color: var(--green); font-weight: 600; }}
  .worse  {{ color: var(--red); font-weight: 600; }}
  .same   {{ color: var(--muted); }}

  /* ── Tabs content ── */
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

  /* ── Contradiction breakdown ── */
  .contradiction-list {{ margin-top: 16px; }}
  .con-item {{
    background: var(--surface2);
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
    border-left: 3px solid var(--border);
  }}
  .con-item.accepted {{ border-left-color: var(--red); }}
  .con-item.detected {{ border-left-color: var(--green); }}
  .con-item.ambiguous {{ border-left-color: var(--amber); }}
  .con-title {{ font-family: var(--font); font-size: 12px; color: var(--muted); margin-bottom: 4px; }}
  .con-lie {{ font-size: 13px; color: var(--text); margin-bottom: 6px; }}
  .con-truth {{ font-size: 12px; color: var(--green); }}
  .con-verdict {{
    display: inline-block; font-size: 11px;
    font-family: var(--font);
    padding: 2px 8px; border-radius: 3px;
    margin-top: 6px;
  }}
  .v-accepted {{ background: rgba(224,92,108,.2); color: var(--red); }}
  .v-detected {{ background: rgba(62,207,142,.15); color: var(--green); }}
  .v-ambiguous {{ background: rgba(244,162,40,.15); color: var(--amber); }}

</style>
</head>
<body>

<div class="header">
  <h1>H-MAD / DST STRESS TEST — EXPERIMENT REPORT</h1>
  <div class="sub">Scenario 1: Adversarial Gaslighting · Generated {timestamp}</div>
  <div class="nav-tabs">
    <button class="nav-tab active" onclick="showTab('overview')">Overview</button>
    <button class="nav-tab" onclick="showTab('transcripts')">Transcripts</button>
    <button class="nav-tab" onclick="showTab('comparison')">Comparison</button>
    <button class="nav-tab" onclick="showTab('charts')">Charts</button>
  </div>
</div>

<div class="container">

  <!-- ── OVERVIEW ── -->
  <div id="tab-overview" class="tab-content active section">
    <div class="section-title">Metric Summary — Per Configuration</div>
    <div class="card-grid">
      {metric_cards}
    </div>
  </div>

  <!-- ── TRANSCRIPTS ── -->
  <div id="tab-transcripts" class="tab-content section">
    <div class="section-title">Full Transcripts with Annotation</div>
    {transcript_sections}
  </div>

  <!-- ── COMPARISON ── -->
  <div id="tab-comparison" class="tab-content section">
    <div class="section-title">Cross-Configuration Comparison</div>
    <div class="card" style="margin-bottom:24px">
      <div class="table-wrap">{comparison_table}</div>
    </div>
  </div>

  <!-- ── CHARTS ── -->
  <div id="tab-charts" class="tab-content section">
    <div class="section-title">Visual Analysis</div>
    <div class="chart-grid">
      <div class="chart-card">
        <div class="chart-title">AHAR — Hallucination Acceptance Rate (lower is better)</div>
        <canvas id="aharChart"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-title">DST Score by Configuration (higher is better)</div>
        <canvas id="dstChart"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-title">Mean Turn-to-Detection (TTD) per Run</div>
        <canvas id="ttdChart"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-title">Detection Rate — % of Contradictions Caught</div>
        <canvas id="detectionChart"></canvas>
      </div>
    </div>
  </div>

</div>

<script>
const CHART_DATA = {chart_data_json};

function showTab(name) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}}

function toggleTranscript(id) {{
  const body = document.getElementById(id);
  const btn = document.getElementById('btn-' + id);
  if (body.style.display === 'none') {{
    body.style.display = 'block';
    btn.textContent = '[ collapse ]';
  }} else {{
    body.style.display = 'none';
    btn.textContent = '[ expand ]';
  }}
}}

const CHART_DEFAULTS = {{
  color: '#7a82a8',
  plugins: {{ legend: {{ labels: {{ color: '#7a82a8', font: {{ family: "'IBM Plex Mono', monospace", size: 11 }} }} }} }},
  scales: {{
    x: {{ ticks: {{ color: '#7a82a8', font: {{ family: "'IBM Plex Mono', monospace", size: 11 }} }}, grid: {{ color: '#2a2f45' }} }},
    y: {{ ticks: {{ color: '#7a82a8', font: {{ family: "'IBM Plex Mono', monospace", size: 11 }} }}, grid: {{ color: '#2a2f45' }} }}
  }}
}};

window.addEventListener('DOMContentLoaded', () => {{
  Chart.defaults.color = '#7a82a8';
  Chart.defaults.borderColor = '#2a2f45';
  Chart.defaults.font.family = "'IBM Plex Mono', monospace";

  const labels = CHART_DATA.labels;
  const colors = ['#5b7cf7', '#3acfcf', '#f4a228', '#3ecf8e'];

  // AHAR Chart
  new Chart(document.getElementById('aharChart'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [
        {{ label: 'AHAR Heuristic', data: CHART_DATA.ahar_heuristic, backgroundColor: 'rgba(91,124,247,0.7)', borderColor: '#5b7cf7', borderWidth: 1 }},
        {{ label: 'AHAR Judge',     data: CHART_DATA.ahar_judge,     backgroundColor: 'rgba(224,92,108,0.7)',  borderColor: '#e05c6c', borderWidth: 1 }}
      ]
    }},
    options: {{ ...CHART_DEFAULTS, plugins: {{ ...CHART_DEFAULTS.plugins }}, scales: {{ ...CHART_DEFAULTS.scales, y: {{ ...CHART_DEFAULTS.scales.y, min: 0, max: 1, ticks: {{ ...CHART_DEFAULTS.scales.y.ticks, callback: v => (v*100)+'%' }} }} }} }}
  }});

  // DST Score Chart
  new Chart(document.getElementById('dstChart'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{ label: 'DST Score /10', data: CHART_DATA.dst_score, backgroundColor: 'rgba(62,207,142,0.7)', borderColor: '#3ecf8e', borderWidth: 1 }}]
    }},
    options: {{ ...CHART_DEFAULTS, scales: {{ ...CHART_DEFAULTS.scales, y: {{ ...CHART_DEFAULTS.scales.y, min: 0, max: 10 }} }} }}
  }});

  // TTD Chart
  new Chart(document.getElementById('ttdChart'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{ label: 'Mean TTD (turns)', data: CHART_DATA.mean_ttd, backgroundColor: 'rgba(244,162,40,0.7)', borderColor: '#f4a228', borderWidth: 1 }}]
    }},
    options: {{ ...CHART_DEFAULTS }}
  }});

  // Detection Rate Chart
  new Chart(document.getElementById('detectionChart'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{ label: 'Detection Rate', data: CHART_DATA.detection_rate, backgroundColor: 'rgba(58,207,207,0.7)', borderColor: '#3acfcf', borderWidth: 1 }}]
    }},
    options: {{ ...CHART_DEFAULTS, scales: {{ ...CHART_DEFAULTS.scales, y: {{ ...CHART_DEFAULTS.scales.y, min: 0, max: 1, ticks: {{ ...CHART_DEFAULTS.scales.y.ticks, callback: v => (v*100)+'%' }} }} }} }}
  }});
}});
</script>
</body>
</html>"""


# ─── Card Rendering ───────────────────────────────────────────────────────────

def _render_metric_card(result: dict) -> str:
    m = result["metrics"]
    config_id = result["config_id"]
    scenario_id = result.get("scenario_id", "")

    dst = m.get("dst_score")
    dst_display = f"{dst:.1f}" if dst and dst >= 0 else "N/A"
    dst_color = (
        "var(--green)" if (dst and dst >= 7) else
        "var(--amber)" if (dst and dst >= 4) else
        "var(--red)"
    )

    ahar_h = m.get("ahar_heuristic", 0)
    ahar_j = m.get("ahar_judge", 0)
    det    = m.get("detection_rate", 0)
    ttd    = m.get("mean_ttd")
    ttd_str = f"{ttd:.1f} turns" if ttd is not None else "—"

    def ahar_class(v): return "good" if v < 0.33 else ("warn" if v < 0.67 else "bad")
    def det_class(v):  return "good" if v > 0.67 else ("warn" if v > 0.33 else "bad")

    reasoning = m.get("judge_reasoning", "")[:300] + ("..." if len(m.get("judge_reasoning","")) > 300 else "")

    badge_class = "b" if config_id == "B" else ""

    con_breakdown = _render_contradiction_breakdown(result)

    return f"""
<div class="card">
  <div class="card-header">
    <div>
      <div class="card-title">Config {config_id} · {scenario_id.replace('_',' ').title()}</div>
      <div style="font-size:12px;color:var(--muted);margin-top:4px;">
        Proponent: <strong style="color:var(--text)">{m.get('proponent_model','?')}</strong>
      </div>
    </div>
    <span class="config-badge {badge_class}">{config_id}</span>
  </div>

  <div class="dst-big" style="color:{dst_color}">{dst_display}</div>
  <div class="dst-label">DST Score / 10</div>

  <div class="score-bar-bg"><div class="score-bar" style="width:{(dst/10*100) if dst and dst>=0 else 0:.0f}%;background:{dst_color}"></div></div>

  <div style="margin-top:20px">
    <div class="metric-row">
      <span class="metric-label">AHAR (heuristic)</span>
      <span class="metric-value {ahar_class(ahar_h)}">{ahar_h:.0%}</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">AHAR (judge)</span>
      <span class="metric-value {ahar_class(ahar_j)}">{ahar_j:.0%}</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Detection Rate</span>
      <span class="metric-value {det_class(det)}">{det:.0%}</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Mean TTD</span>
      <span class="metric-value {'na' if ttd is None else 'warn'}">{ttd_str}</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Contradictions injected</span>
      <span class="metric-value">{m.get('total_injected', 0)}</span>
    </div>
  </div>

  <div class="model-row">
    <span class="model-tag">Proponent: <span>{m.get('proponent_model','?')}</span></span>
    <span class="model-tag">Gaslighter: <span>{m.get('gaslighter_model','?')}</span></span>
    <span class="model-tag">Judge: <span>{m.get('judge_model','?')}</span></span>
  </div>

  {f'<div class="reasoning">{reasoning}</div>' if reasoning else ''}

  {con_breakdown}
</div>"""


def _render_contradiction_breakdown(result: dict) -> str:
    m = result["metrics"]
    per_con = m.get("ahar_per_contradiction", [])
    if not per_con:
        return ""

    items = []
    for c in per_con:
        judge_resp = c.get("judge_response", "unknown")
        verdict_class = (
            "accepted" if judge_resp == "accepted" else
            "detected" if judge_resp == "pushed_back" else
            "ambiguous"
        )
        verdict_badge_class = (
            "v-accepted" if judge_resp == "accepted" else
            "v-detected" if judge_resp == "pushed_back" else
            "v-ambiguous"
        )
        verdict_label = {
            "accepted": "✗ Accepted",
            "pushed_back": "✓ Pushed Back",
            "ambiguous": "~ Ambiguous",
        }.get(judge_resp, judge_resp)

        items.append(f"""
<div class="con-item {verdict_class}">
  <div class="con-title">Injected at turn {c.get('injection_turn','?')}</div>
  <div class="con-lie">Lie: {c.get('planted_lie','')}</div>
  <div class="con-truth">Truth: {c.get('truth','')}</div>
  <span class="con-verdict {verdict_badge_class}">{verdict_label}</span>
</div>""")

    return f'<div class="contradiction-list" style="margin-top:16px;border-top:1px solid var(--border);padding-top:16px">{"".join(items)}</div>'


# ─── Transcript Rendering ─────────────────────────────────────────────────────

def _render_transcript_section(result: dict) -> str:
    transcript = result.get("transcript", [])
    config_id  = result["config_id"]
    scenario_id = result.get("scenario_id", "")
    deployed   = {c["turn"]: c for c in result.get("deployed_contradictions", [])}
    det_turns  = set(result.get("proponent_detection_turns", []))
    uid = f"transcript-{config_id}-{scenario_id}"

    rows = []
    for msg in transcript:
        turn  = msg.get("turn", "?")
        role  = msg.get("role", "user")
        phase = msg.get("phase", "debate")
        content = msg.get("content", "").replace("<", "&lt;").replace(">", "&gt;")

        extra_class = ""
        badge = ""
        if turn in deployed and role == "user":
            extra_class = "contradiction"
            lie = deployed[turn].get("planted_lie", "")
            badge = f'<div><span class="contradiction-badge">💥 contradiction injected</span></div>'

        if turn in det_turns and role == "assistant":
            extra_class = "detected"
            badge = f'<div><span class="detection-badge">✓ detection signal</span></div>'

        if phase == "setup":
            extra_class += " setup"
            role_badge = '<span class="msg-role setup">SETUP</span>'
        elif role == "user":
            role_badge = '<span class="msg-role user">USER</span>'
        else:
            role_badge = '<span class="msg-role assistant">ASST</span>'

        rows.append(f"""
<div class="msg {extra_class}">
  <div class="msg-turn">T{turn}</div>
  <div>{role_badge}</div>
  <div class="msg-content">{badge}{content}</div>
</div>""")

    header_label = f"Config {config_id} · {scenario_id.replace('_',' ').title()} · {len(transcript)} turns"
    return f"""
<div class="transcript-card">
  <div class="transcript-header">
    <span class="title">{header_label}</span>
    <button class="transcript-toggle" id="btn-{uid}" onclick="toggleTranscript('{uid}')">[ expand ]</button>
  </div>
  <div class="transcript-body" id="{uid}" style="display:none">
    {"".join(rows)}
  </div>
</div>"""


# ─── Comparison Table ─────────────────────────────────────────────────────────

def _render_comparison_table(results: list[dict]) -> str:
    if not results:
        return "<p style='color:var(--muted)'>No results.</p>"

    rows = []
    for r in results:
        m = r["metrics"]
        dst   = m.get("dst_score")
        ahar_j = m.get("ahar_judge", 0)
        det   = m.get("detection_rate", 0)
        ttd   = m.get("mean_ttd")

        rows.append(f"""
<tr>
  <td>Config {r['config_id']}</td>
  <td>{r.get('scenario_id','').replace('_',' ')}</td>
  <td>{m.get('proponent_model','?')}</td>
  <td>{m.get('gaslighter_model','?')}</td>
  <td class="{'better' if ahar_j < 0.34 else ('worse' if ahar_j > 0.66 else '')}">{ahar_j:.0%}</td>
  <td class="{'better' if det > 0.66 else ('worse' if det < 0.34 else '')}">{det:.0%}</td>
  <td class="{'better' if (ttd is not None and ttd <= 2) else ''}">{'—' if ttd is None else f'{ttd:.1f}'}</td>
  <td class="{'better' if (dst and dst >= 7) else ('worse' if (dst and dst < 4) else '')}">{f'{dst:.1f}' if dst and dst >= 0 else '—'}</td>
</tr>""")

    return f"""
<table>
  <thead>
    <tr>
      <th>Config</th>
      <th>Scenario</th>
      <th>Proponent</th>
      <th>Gaslighter</th>
      <th>AHAR (Judge) ↓</th>
      <th>Det. Rate ↑</th>
      <th>Mean TTD ↓</th>
      <th>DST Score ↑</th>
    </tr>
  </thead>
  <tbody>{"".join(rows)}</tbody>
</table>"""


# ─── Chart Data ───────────────────────────────────────────────────────────────

def _build_chart_data(results: list[dict]) -> dict:
    labels = []
    ahar_h, ahar_j, dst_scores, ttd_vals, det_rates = [], [], [], [], []

    for r in results:
        m = r["metrics"]
        label = f"Config {r['config_id']} / {r.get('scenario_id','')[:8]}"
        labels.append(label)
        ahar_h.append(round(m.get("ahar_heuristic", 0), 3))
        ahar_j.append(round(m.get("ahar_judge", 0), 3))
        dst = m.get("dst_score")
        dst_scores.append(round(dst, 1) if dst and dst >= 0 else 0)
        ttd = m.get("mean_ttd")
        ttd_vals.append(round(ttd, 1) if ttd is not None else 0)
        det_rates.append(round(m.get("detection_rate", 0), 3))

    return {
        "labels": labels,
        "ahar_heuristic": ahar_h,
        "ahar_judge": ahar_j,
        "dst_score": dst_scores,
        "mean_ttd": ttd_vals,
        "detection_rate": det_rates,
    }
