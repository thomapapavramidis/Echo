"""Intentionally plain dashboard for observing the RunPod proof of concept."""

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Compute Echo POC</title><style>
body{font:14px system-ui;margin:20px;background:#111827;color:#e5e7eb}h1{margin-bottom:4px}
.warning{background:#7c2d12;padding:12px;border:2px solid #fb923c}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin-top:12px}
.card{border:1px solid #4b5563;background:#1f2937;padding:12px}.badge{display:inline-block;background:#374151;border:1px solid #9ca3af;padding:4px 7px;margin:3px}
.helper{border-color:#f59e0b}.sequence span{display:inline-block;padding:3px 6px;margin:2px;background:#374151}.active{outline:2px solid #60a5fa}
button,input{padding:8px;margin:4px}button{cursor:pointer}canvas{width:100%;height:180px;background:#030712}.value{font-size:22px;font-weight:700}.muted{color:#9ca3af}pre{white-space:pre-wrap;max-height:240px;overflow:auto}
</style></head><body>
<h1>Compute Echo — RunPod proof of concept</h1>
<div class="warning"><strong>HOST GPU TELEMETRY — PROOF OF CONCEPT</strong><br>
NVML values come from the GPU hosts. They are not an auditor-controlled external meter and cannot satisfy the production physical-evidence trust gate.</div>
<p><input id="token" type="password" size="44" placeholder="ECHO_ADMIN_TOKEN (kept in this tab only)"><button onclick="refresh()">Connect</button></p>
<div class="grid">
 <div class="card"><h2>Experiment control</h2><div>Current route: <span class="value" id="route">unknown</span></div>
  <button onclick="route('local')">Route to declared site</button><button onclick="route('forwarded')">Forward to helper</button>
  <button onclick="run()">Start fresh epoch</button><button onclick="transition()">Transition to selected route</button>
  <select id="target"><option value="local">local</option><option value="forwarded">forwarded</option></select>
  <p class="muted">Controls call the declared-site worker. They never write telemetry or detector output.</p></div>
 <div class="card"><h2>Evidence</h2><div>Correctness: <span id="correct">PENDING</span></div><div>Deadline: <span id="deadline">PENDING</span></div>
  <div>Progress: <span id="progress">0/0</span></div><div>Production physical response: <strong id="physical">INSUFFICIENT</strong></div>
  <div>Host diagnostic: <span id="hostresult">NO_HOST_TELEMETRY</span></div></div>
</div>
<div class="card"><h2>Expected challenge sequence</h2><div class="sequence" id="sequence">No epoch scheduled</div></div>
<div class="grid">
 <div class="card"><h2>Declared-site GPU</h2><span class="badge">HOST GPU TELEMETRY — PROOF OF CONCEPT</span><span class="badge">DECLARED-SITE HOST DIAGNOSTIC</span>
  <div id="siteNow">No samples</div><canvas id="siteChart" width="600" height="180"></canvas></div>
 <div class="card helper"><h2>Remote-helper GPU</h2><span class="badge">HOST GPU TELEMETRY — PROOF OF CONCEPT</span><span class="badge">ATTACK GROUND TRUTH — NOT NORMALLY AVAILABLE TO AUDITOR</span>
  <div id="helperNow">No samples</div><canvas id="helperChart" width="600" height="180"></canvas></div>
</div><div class="card"><h2>Raw state</h2><pre id="raw"></pre></div>
<script>
let timer; const auth=()=>({'Authorization':'Bearer '+document.getElementById('token').value});
async function call(path,options={}){options.headers={...auth(),'Content-Type':'application/json'};let r=await fetch(path,options);if(!r.ok)throw Error(await r.text());return r.json()}
function latest(rows){for(let i=rows.length-1;i>=0;i--)if(rows[i].sample&&rows[i].sample.available)return rows[i].sample;return null}
function draw(id,rows){let c=document.getElementById(id),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);let a=rows.filter(o=>o.sample&&o.sample.available).slice(-120);if(a.length<2)return;
 let powers=a.map(o=>o.sample.board_or_module_power_draw_w),max=Math.max(1,...powers),min=Math.min(...powers);x.strokeStyle='#fb923c';x.beginPath();powers.forEach((v,i)=>{let px=i/(a.length-1)*c.width,py=c.height-10-(v-min)/Math.max(1,max-min)*(c.height-20);i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();
 x.strokeStyle='#60a5fa';x.beginPath();a.map(o=>o.sample.gpu_compute_utilization_pct).forEach((v,i)=>{let px=i/(a.length-1)*c.width,py=c.height-10-v/100*(c.height-20);i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()}
function now(id,s){document.getElementById(id).textContent=s?`${s.gpu_model} | ${s.board_or_module_power_draw_w.toFixed(1)} W | compute ${s.gpu_compute_utilization_pct}% | memory ${s.gpu_memory_utilization_pct}% | ${s.collector_backend}`:'No fresh available sample'}
async function refresh(){try{let s=await call('/demo/state');document.getElementById('route').textContent=s.route;document.getElementById('correct').textContent=s.computational_correctness;document.getElementById('deadline').textContent=s.deadline_status;
 let r=s.runner,n=r.completed_segments||0,t=r.total_segments||0;document.getElementById('progress').textContent=`${n}/${t}`;document.getElementById('physical').textContent=s.local_physical_response.production_physical_response;document.getElementById('hostresult').textContent=s.local_physical_response.result;
 let q=s.expected_challenge_sequence||[],at=r.current_challenge_index;document.getElementById('sequence').innerHTML=q.length?q.map((v,i)=>`<span class="${i===at?'active':''}">${i+1} ${v}</span>`).join(''):'No epoch scheduled';
 let a=s.telemetry.declared_site,b=s.telemetry.remote_helper;now('siteNow',latest(a));now('helperNow',latest(b));draw('siteChart',a);draw('helperChart',b);document.getElementById('raw').textContent=JSON.stringify(s,null,2);clearTimeout(timer);timer=setTimeout(refresh,500)}catch(e){document.getElementById('raw').textContent=e}}
async function route(v){await call('/demo/route/'+v,{method:'POST'});document.getElementById('target').value=v;refresh()}
async function run(){await call('/demo/run',{method:'POST'});refresh()}
async function transition(){await call('/demo/transition/'+document.getElementById('target').value,{method:'POST'});refresh()}
</script></body></html>"""
