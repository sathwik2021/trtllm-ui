const app = document.getElementById("app");
let state = {settings:null, caps:null, models:[], deployments:[], selectedDeployment:null};

async function api(url, options={}) {
  const r = await fetch(url, {headers:{"Content-Type":"application/json"}, ...options});
  const text = await r.text();
  let data; try { data = text ? JSON.parse(text) : {}; } catch { data = text; }
  if (!r.ok) throw new Error(data.detail || text || r.statusText);
  return data;
}
function esc(s){return String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function fmt(n){return n==null?"":(Number(n)/1024/1024/1024).toFixed(2)+" GB";}
async function refresh(){
  state.settings=await api("/api/settings");
  state.caps=await api("/api/capabilities");
  state.models=await api("/api/models");
  state.deployments=await api("/api/deployments");
}
function nav(){document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>show(b.dataset.page));}
async function show(page){
  try{await refresh();}catch(e){app.innerHTML=`<div class="card error">${esc(e.message)}</div>`;return;}
  if(page==="dashboard") dashboard();
  else if(page==="models") models();
  else if(page==="deploy") deploy();
  else if(page==="deployments") deployments();
  else if(page==="benchmark") benchmark();
  else if(page==="chat") chat();
  else if(page==="storage") storage();
  else if(page==="profiles") profiles();
  else if(page==="diagnostics") diagnostics();
}
function capParsed(){return state.caps?.manifest?.parsed || null;}
function precisionBadges(precision){
  if(!precision) return "";
  return Object.entries(precision).map(([k,v])=>{
    const sup=v.supported;
    const cls=sup===true?"success":sup===false?"muted":"warn";
    const label=sup===true?"available":sup===false?"unavailable (hardware)":"unknown";
    return `<span class="${cls}" title="${esc(v.note||"")}">${k.toUpperCase()}: ${label}</span>`;
  }).join(" &nbsp; ");
}
function dashboard(){
  const c=state.caps;
  const g=c?.manifest?.commands?.[0]?.stdout || "";
  const parsed=capParsed();
  const gpu=parsed?.gpu?.gpus?.[0];
  app.innerHTML=`<div class="card"><h2>Dashboard</h2>
    <p>Control panel: <b>${esc(location.origin)}</b></p>
    <p>Model directory: <code>${esc(state.settings.model_dir)}</code></p>
    <p>Data directory: <code>${esc(state.settings.data_dir)}</code></p>
    ${state.settings.path_warnings?.length?`<div class="danger">${state.settings.path_warnings.map(esc).join("<br>")}</div>`:""}
    <h3>Diagnostics</h3>
    ${c.status==="not_yet_probed"?`<p class="muted">Not yet probed — capability-aware checks (precision, kv_cache_dtype options, VRAM pre-flight) stay disabled until you probe once.</p>`:`<p class="success">Last probe: ${esc(c.manifest.probed_at)}</p>`}
    ${gpu?`<p><b>${esc(gpu.name)}</b> — ${esc(String(gpu.vram_total_mb))}MB total, ${esc(String(gpu.vram_free_mb))}MB free at probe time, compute capability ${esc(String(gpu.compute_capability))}, ${esc(String(parsed.gpu.count))} GPU(s) detected</p>
    <p>${precisionBadges(parsed.precision)}</p>`:""}
    ${c.status!=="not_yet_probed"?`<pre>${esc(g)}</pre>`:""}
    <button onclick="show('diagnostics')">Run Diagnostics</button>
  </div>
  <div class="card"><h3>Deployments</h3>${state.deployments.map(d=>`<p><b>${esc(d.deployment_id)}</b> — ${esc(d.status)} — port ${esc(d.port)}</p>`).join("")||"None"}</div>`;
}
function models(){
  app.innerHTML=`<div class="card"><h2>Models</h2><table><tr><th>Name</th><th>Size</th><th>Architecture</th><th>Path</th><th></th></tr>
  ${state.models.map(m=>`<tr><td>${esc(m.name)}</td><td>${fmt(m.size_bytes)}</td><td>${esc((m.architectures||[]).join(", "))}</td><td><code>${esc(m.host_path)}</code></td><td><button onclick="prepareDeploy('${esc(m.name)}')">Deploy</button></td></tr>`).join("")}</table></div>`;
}
function prepareDeploy(name){show("deploy").then(()=>{document.getElementById("model").value=name; document.getElementById("served").value=name; updateCommand();});}
function deploy(){
  const options=state.models.map(m=>`<option>${esc(m.name)}</option>`).join("");
  const parsed=capParsed();
  const kvOptions=parsed?.kv_cache_dtype_options||null;
  const parallelMax=parsed?.parallel_max;
  // Capability-driven: parallelism dimensions are moot with <=1 detected GPU
  // (or unknown, if never probed) -- hide the controls rather than show
  // disabled/default-1 selectors nobody on this hardware can use.
  const showParallelism = parallelMax!=null && parallelMax>1;
  const kvField = kvOptions
    ? `<select id="kv" onchange="updateCommand()"><option value="">(unset)</option>${kvOptions.map(o=>`<option>${esc(o)}</option>`).join("")}</select>
       <p class="muted">Options limited to what the last capability probe confirmed this GPU/image can execute.</p>`
    : `<input id="kv" oninput="updateCommand()"><p class="muted">Run a capability probe (Diagnostics page) to restrict this list to hardware-supported values.</p>`;
  app.innerHTML=`<div class="card"><h2>Deploy</h2>
  <label>Model<select id="model" onchange="document.getElementById('served').value=this.value;updateCommand()">${options}</select></label>
  <label>Backend<input id="backend" value="pytorch" oninput="updateCommand()"></label>
  <label>Internal TensorRT-LLM host<input id="host" value="0.0.0.0" oninput="updateCommand()"></label>
  <label>Host-published port<input id="port" type="number" value="8000" oninput="updateCommand()"></label>
  <label>Network exposure
    <select id="pubhost" onchange="updateCommand()">
      <option value="127.0.0.1" selected>This machine only (127.0.0.1) — recommended</option>
      <option value="0.0.0.0">Local network (0.0.0.0) — no authentication on this server, anyone on your network can reach it</option>
    </select>
  </label>
  <label>Served model name<input id="served" oninput="updateCommand()"></label>
  <label>Max batch size<input id="mb" type="number" oninput="updateCommand()"></label>
  <label>Max input/context length<input id="ms" type="number" oninput="updateCommand()"></label>
  <label>Max output tokens <span class="muted">(VRAM estimate only — not a trtllm-serve flag)</span><input id="mot" type="number" oninput="updateCommand()"></label>
  <label>KV cache dtype${kvField}</label>
  <label>Free GPU memory fraction<input id="fg" type="number" step="0.01" oninput="updateCommand()"></label>
  ${showParallelism?`<details open><summary>Parallelism (${parallelMax} GPUs detected)</summary>
    <label>Tensor parallel size<input id="tp" type="number" value="1" min="1" max="${parallelMax}" oninput="updateCommand()"></label>
    <label>Pipeline parallel size<input id="pp" type="number" value="1" min="1" max="${parallelMax}" oninput="updateCommand()"></label>
    <label>Context parallel size<input id="cp" type="number" value="1" min="1" max="${parallelMax}" oninput="updateCommand()"></label>
    <label>MoE expert parallel size<input id="ep" type="number" value="1" min="1" max="${parallelMax}" oninput="updateCommand()"></label>
    <label>GPUs per node<input id="gpn" type="number" oninput="updateCommand()"></label>
  </details>`:`<p class="muted">Parallelism controls hidden — ${parallelMax==null?"run a capability probe to detect GPU count":"only "+parallelMax+" GPU detected, tensor/pipeline/context/expert parallelism need >1"}.</p>`}
  <details><summary>Expert</summary>
    <label>Custom module dirs<input id="cmdirs" oninput="updateCommand()"></label>
    <label><input id="trc" type="checkbox" onchange="updateCommand()"> trust_remote_code</label>
    <label>Unsafe confirmation<input id="ack" placeholder="Type ENABLE UNSAFE" oninput="updateCommand()"></label>
    <label>Extra flags JSON<textarea id="extra" oninput="updateCommand()">{}</textarea></label>
    <label>Extra LLM API options (nested config, mounted as --config)
      <textarea id="extraLlmOpts" oninput="updateCommand()" placeholder='e.g. {"cuda_graph_config": {"enable_padding": true}}'></textarea>
      <p class="muted">Unconfirmed against this exact trtllm-serve build — for config fields not exposed as top-level flags (see LLM Args in deployment logs). Written as JSON to a mounted file. Leave blank to skip.</p>
    </label>
  </details>
  <h3>Generated command</h3><pre id="command"></pre>
  <button onclick="runPreflight()">Run Pre-flight Check</button>
  <button onclick="launch()">Deploy</button>
  <div id="preflight"></div>
  <div id="deploymsg"></div></div>`;
  if(state.models[0]){document.getElementById("model").value=state.models[0].name;document.getElementById("served").value=state.models[0].name;}
  updateCommand();
}
function configFromForm(){
  let extra={}; try{extra=JSON.parse(document.getElementById("extra").value||"{}")}catch{}
  let extraLlmOpts=null;
  const rawOpts=(document.getElementById("extraLlmOpts")?.value||"").trim();
  if(rawOpts){try{extraLlmOpts=JSON.parse(rawOpts)}catch{}}
  const num=id=>{const el=document.getElementById(id);if(!el)return null;const v=el.value;return v===""?null:Number(v)};
  return {model_name:document.getElementById("model").value,backend:document.getElementById("backend").value,host:document.getElementById("host").value,port:Number(document.getElementById("port").value),
    publish_host:document.getElementById("pubhost").value,
    served_model_name:document.getElementById("served").value,max_batch_size:num("mb"),max_seq_len:num("ms"),max_output_tokens:num("mot"),tensor_parallel_size:num("tp")||1,
    pipeline_parallel_size:num("pp")||1,context_parallel_size:num("cp")||1,moe_expert_parallel_size:num("ep")||1,gpus_per_node:num("gpn"),
    kv_cache_dtype:document.getElementById("kv").value||null,free_gpu_memory_fraction:num("fg"),trust_remote_code:document.getElementById("trc").checked,
    custom_module_dirs:document.getElementById("cmdirs").value||null,unsafe_ack:document.getElementById("ack").value||null,extra_flags:extra,
    extra_llm_api_options:extraLlmOpts};
}
async function runPreflight(){
  const el=document.getElementById("preflight"); el.innerHTML="<p class=\"muted\">Running pre-flight checks…</p>";
  try{
    const r=await api("/api/deployments/preflight",{method:"POST",body:JSON.stringify(configFromForm())});
    const rows=r.checks.map(c=>`<p class="${c.status==='pass'?'success':c.status==='warn'?'warn':'error'}">${c.status==='pass'?'✓':c.status==='warn'?'⚠':'✗'} <b>${esc(c.name)}</b> — ${esc(c.detail)}</p>`).join("");
    const vram=r.vram_estimate?`<p class="muted">VRAM estimate (heuristic, approximate — not exact): ~${esc(String(r.vram_estimate.total_estimated_mb))}MB total (${esc(String(r.vram_estimate.weights_mb))}MB weights + ${esc(String(r.vram_estimate.kv_cache_mb))}MB KV cache, method: ${esc(r.vram_estimate.method)})</p>`:"";
    el.innerHTML=`<div class="card">${r.feasible?'<p class="success"><b>Configuration appears feasible.</b></p>':'<p class="error"><b>Configuration has blocking issues — see below.</b></p>'}${rows}${vram}<p class="muted">Advisory only — the real Docker/GPU/trtllm-serve stack is the final authority.</p></div>`;
  }catch(e){el.innerHTML=`<p class="error">${esc(e.message)}</p>`;}
}
function updateCommand(){
  const el=document.getElementById("command"); if(!el)return;
  const c=configFromForm(); const model=state.models.find(m=>m.name===c.model_name);
  if(!model){el.textContent="Select a model";return;}
  let a=["docker","run","-d","--name","trtllm-ui-PREVIEW","--gpus","all","--ipc=host","--ulimit","memlock=-1","--ulimit","stack=67108864","-p",`${c.publish_host}:${c.port}:${c.port}`,"-v",`${model.host_path}:${model.container_path}:ro`];
  if(c.extra_llm_api_options) a.push("-v","<data_dir>/llm_api_options/<id>.yaml:/trtllm_extra_config.yaml:ro");
  a.push(state.settings.docker_image,"trtllm-serve","serve",model.container_path,"--backend",c.backend,"--host",c.host,"--port",String(c.port),"--served_model_name",c.served_model_name||c.model_name);
  [["max_batch_size",c.max_batch_size],["max_seq_len",c.max_seq_len],["tensor_parallel_size",c.tensor_parallel_size!==1?c.tensor_parallel_size:null],["pipeline_parallel_size",c.pipeline_parallel_size!==1?c.pipeline_parallel_size:null],["context_parallel_size",c.context_parallel_size!==1?c.context_parallel_size:null],["moe_expert_parallel_size",c.moe_expert_parallel_size!==1?c.moe_expert_parallel_size:null],["gpus_per_node",c.gpus_per_node],["kv_cache_dtype",c.kv_cache_dtype],["free_gpu_memory_fraction",c.free_gpu_memory_fraction]].forEach(([k,v])=>{if(v!=null)a.push("--"+k,String(v));});
  if(c.trust_remote_code)a.push("--trust_remote_code"); if(c.custom_module_dirs)a.push("--custom_module_dirs",c.custom_module_dirs);
  Object.entries(c.extra_flags||{}).forEach(([k,v])=>{if(v===true)a.push(k.startsWith("--")?k:"--"+k);else if(v!==false&&v!=null&&v!=="")a.push(k.startsWith("--")?k:"--"+k,String(v));});
  if(c.extra_llm_api_options) a.push("--config","/trtllm_extra_config.yaml");
  el.textContent=a.map(x=>/[\s"'\\]/.test(x)?`'${x.replaceAll("'","'\\''")}'`:x).join(" ");
}
async function launch(){
  try{const d=await api("/api/deployments",{method:"POST",body:JSON.stringify(configFromForm())});document.getElementById("deploymsg").innerHTML=`<p class="success">Started ${esc(d.deployment_id)}.</p>`;show("deployments");}
  catch(e){document.getElementById("deploymsg").innerHTML=`<p class="error">${esc(e.message)}</p>`;}
}
function deployments(){
  app.innerHTML=`<div class="card"><h2>Deployments</h2>${state.deployments.map(d=>`<div class="card"><b>${esc(d.deployment_id)}</b> — ${esc(d.status)} — ${esc(d.reason||"")} — port ${esc(d.port)}<br><button onclick="logs('${esc(d.deployment_id)}')">Logs</button> <button onclick="stop('${esc(d.deployment_id)}')">Stop</button></div>`).join("")||"None"}</div>`;
}
async function stop(id){await api(`/api/deployments/${id}/stop`,{method:"POST"});show("deployments");}
function logs(id){app.innerHTML=`<div class="card"><h2>Logs: ${esc(id)}</h2><pre id="logs"></pre></div>`;const es=new EventSource(`/api/deployments/${id}/logs`);es.onmessage=e=>document.getElementById("logs").textContent+=JSON.parse(e.data)+"\n";}
function chat(){
  app.innerHTML=`<div class="card"><h2>Chat / Test</h2><label>Deployment<select id="chatdep">${state.deployments.filter(d=>d.status==="running"||d.status==="ready").map(d=>`<option value="${d.port}">${esc(d.deployment_id)}:${d.port}</option>`).join("")}</select></label>
  <label>Message<textarea id="msg">Hello! Explain what you are in one sentence.</textarea></label><label>Temperature<input id="temp" type="number" step=".1" value=".7"></label><label>Top P<input id="top" type="number" step=".1" value="1"></label><label>Max tokens<input id="mt" type="number" value="128"></label><button onclick="sendChat()">Send</button><pre id="chatout"></pre></div>`;
}
async function sendChat(){
  const port=document.getElementById("chatdep").value;
  const body={messages:[{role:"user",content:document.getElementById("msg").value}],temperature:Number(document.getElementById("temp").value),top_p:Number(document.getElementById("top").value),max_tokens:Number(document.getElementById("mt").value)};
  const t=performance.now();try{const r=await fetch(`http://127.0.0.1:${port}/v1/chat/completions`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});const x=await r.json();document.getElementById("chatout").textContent=JSON.stringify({latency_ms:Math.round(performance.now()-t),response:x},null,2);}catch(e){document.getElementById("chatout").textContent=e.message;}
}
async function storage(){
  const s=await api("/api/storage");app.innerHTML=`<div class="card"><h2>Storage</h2><pre>${esc(JSON.stringify(s,null,2))}</pre></div>`;
}
async function profiles(){
  const p=await api("/api/profiles");app.innerHTML=`<div class="card"><h2>Profiles</h2><input id="pname" placeholder="profile name"><button onclick="saveProfile()">Save current Deploy form</button><ul>${p.map(n=>`<li>${esc(n)} <button onclick="loadProfile('${esc(n)}')">Load</button> <button onclick="deleteProfile('${esc(n)}')">Delete</button></li>`).join("")}</ul></div>`;
}
async function saveProfile(){alert("Open Deploy, configure the desired deployment, then use this page after wiring the current form state. Profile API is implemented; UI persistence is intentionally kept minimal in this first runnable bundle.");}
async function loadProfile(n){const x=await api(`/api/profiles/${encodeURIComponent(n)}`);alert("Profile loaded: "+n);console.log(x);}
async function deleteProfile(n){await api(`/api/profiles/${encodeURIComponent(n)}`,{method:"DELETE"});profiles();}
async function diagnostics(){
  app.innerHTML=`<div class="card"><h2>Diagnostics</h2><button onclick="runProbe()">Run Diagnostics</button><div id="diag"></div></div>`;
  const c=state.caps;if(c.manifest) renderDiag(c.manifest);
}
async function runProbe(){await api("/api/capabilities/probe",{method:"POST"});document.getElementById("diag").innerHTML="<p>Probe started. Refresh in a few seconds.</p>";setTimeout(async()=>{state.caps=await api("/api/capabilities");if(state.caps.manifest)renderDiag(state.caps.manifest);},5000);}
function renderDiag(m){document.getElementById("diag").innerHTML=m.commands.map((c,i)=>`<details open="${i===3}"><summary>${esc(c.argv.join(" "))} — exit ${c.returncode}</summary><pre>${esc((c.stdout||"")+"\n"+(c.stderr||""))}</pre></details>`).join("");}

async function benchmark(){
  const running=state.deployments.filter(d=>d.status==="running"||d.status==="ready");
  const history=await api("/api/benchmarks");
  app.innerHTML=`<div class="card"><h2>Benchmark</h2>
  <label>Deployment<select id="bdep">${running.map(d=>`<option value="${d.port}" data-model="${esc((d.config||{}).served_model_name||d.deployment_id)}">${esc(d.deployment_id)}:${d.port}</option>`).join("")||`<option value="">No running deployments</option>`}</select></label>
  <label>Request count<input id="brc" type="number" value="20"></label>
  <label>Concurrency<input id="bcc" type="number" value="1"></label>
  <label>Max tokens per request<input id="bmt" type="number" value="256"></label>
  <label>Prompt (optional)<textarea id="bprompt" placeholder="Leave blank for default prompt"></textarea></label>
  <button onclick="runBenchmark()" ${running.length?"":"disabled"}>Run Benchmark</button>
  <div id="bstatus"></div>
  <div id="bresult"></div></div>
  <div class="card"><h3>Past runs</h3><div id="bhistory">${renderBenchmarkHistory(history)}</div></div>`;
}
function renderBenchmarkHistory(list){
  if(!list.length) return "<p class=\"muted\">None yet.</p>";
  return list.map(r=>`<div class="card">
    <b>${esc(new Date(r.created_at*1000).toLocaleString())}</b> — ${esc(r.config.served_model_name)} — ${esc(String(r.config.request_count))} req @ concurrency ${esc(String(r.config.concurrency))}<br>
    throughput: ${r.throughput_tokens_per_s!=null?r.throughput_tokens_per_s.toFixed(1)+" tok/s":"n/a"} — p50: ${r.latency_s.p50!=null?r.latency_s.p50.toFixed(2)+"s":"n/a"} — p95: ${r.latency_s.p95!=null?r.latency_s.p95.toFixed(2)+"s":"n/a"} — failed: ${esc(String(r.requests_failed))}
    <br><button onclick="viewBenchmark('${esc(r.id)}')">View</button> <button onclick="deleteBenchmark('${esc(r.id)}')">Delete</button>
  </div>`).join("");
}
async function runBenchmark(){
  const sel=document.getElementById("bdep");
  const port=Number(sel.value);
  const served_model_name=sel.selectedOptions[0]?.dataset.model;
  const body={host:"127.0.0.1",port,served_model_name,
    request_count:Number(document.getElementById("brc").value),
    concurrency:Number(document.getElementById("bcc").value),
    max_tokens:Number(document.getElementById("bmt").value),
    prompt:document.getElementById("bprompt").value||null};
  const statusEl=document.getElementById("bstatus");
  statusEl.innerHTML="<p class=\"muted\">Starting benchmark…</p>";
  let job;
  try{job=await api("/api/benchmarks",{method:"POST",body:JSON.stringify(body)});}
  catch(e){statusEl.innerHTML=`<p class="error">${esc(e.message)}</p>`;return;}
  statusEl.innerHTML="<p class=\"muted\">Running… this can take a while depending on request count/concurrency.</p>";
  const poll=async()=>{
    const s=await api("/api/benchmarks/status");
    if(s.status==="running"){setTimeout(poll,2000);return;}
    if(s.status==="error"){statusEl.innerHTML=`<p class="error">Benchmark failed: ${esc(s.error||"unknown error")}</p>`;return;}
    statusEl.innerHTML="<p class=\"success\">Done.</p>";
    viewBenchmark(job.id);
    state_refresh_history();
  };
  setTimeout(poll,1500);
}
async function state_refresh_history(){const h=await api("/api/benchmarks");const el=document.getElementById("bhistory");if(el)el.innerHTML=renderBenchmarkHistory(h);}
async function viewBenchmark(id){
  const r=await api(`/api/benchmarks/${id}`);
  document.getElementById("bresult").innerHTML=`<div class="card">
    <h3>Result</h3>
    <p>Model: ${esc(r.config.served_model_name)} — ${esc(String(r.config.request_count))} requests @ concurrency ${esc(String(r.config.concurrency))}, max_tokens ${esc(String(r.config.max_tokens))}</p>
    <p>Wall time: ${r.wall_time_s.toFixed(2)}s — Throughput: ${r.throughput_tokens_per_s!=null?r.throughput_tokens_per_s.toFixed(1)+" tok/s":"n/a"}</p>
    <p>OK: ${esc(String(r.requests_ok))} — Failed: ${esc(String(r.requests_failed))}${r.errors.length?" — e.g. "+esc(r.errors[0]):""}</p>
    <p>Latency — min: ${fmtS(r.latency_s.min)} avg: ${fmtS(r.latency_s.avg)} p50: ${fmtS(r.latency_s.p50)} p95: ${fmtS(r.latency_s.p95)} max: ${fmtS(r.latency_s.max)}</p>
    <p>GPU utilization % — min: ${fmtN(r.gpu.utilization_gpu_pct.min)} avg: ${fmtN(r.gpu.utilization_gpu_pct.avg)} max: ${fmtN(r.gpu.utilization_gpu_pct.max)}</p>
    <p>GPU memory used (MB) — min: ${fmtN(r.gpu.memory_used_mb.min)} avg: ${fmtN(r.gpu.memory_used_mb.avg)} max: ${fmtN(r.gpu.memory_used_mb.max)}</p>
    <p class="muted">${esc(String(r.gpu.samples_captured))} GPU samples captured during run.</p>
  </div>`;
}
function fmtS(v){return v==null?"n/a":v.toFixed(2)+"s";}
function fmtN(v){return v==null?"n/a":v.toFixed(1);}
async function deleteBenchmark(id){await api(`/api/benchmarks/${id}`,{method:"DELETE"});state_refresh_history();}

nav();show("dashboard");
