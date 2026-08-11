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
  else if(page==="chat") chat();
  else if(page==="storage") storage();
  else if(page==="profiles") profiles();
  else if(page==="diagnostics") diagnostics();
}
function dashboard(){
  const c=state.caps;
  const g=c?.manifest?.commands?.[0]?.stdout || "";
  app.innerHTML=`<div class="card"><h2>Dashboard</h2>
    <p>Control panel: <b>${esc(location.origin)}</b></p>
    <p>Model directory: <code>${esc(state.settings.model_dir)}</code></p>
    <p>Data directory: <code>${esc(state.settings.data_dir)}</code></p>
    ${state.settings.path_warnings?.length?`<div class="danger">${state.settings.path_warnings.map(esc).join("<br>")}</div>`:""}
    <h3>Diagnostics</h3>
    ${c.status==="not_yet_probed"?`<p class="muted">Not yet probed.</p>`:`<p class="success">Last probe: ${esc(c.manifest.probed_at)}</p><pre>${esc(g)}</pre>`}
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
  app.innerHTML=`<div class="card"><h2>Deploy</h2>
  <label>Model<select id="model" onchange="document.getElementById('served').value=this.value;updateCommand()">${options}</select></label>
  <label>Backend<input id="backend" value="pytorch" oninput="updateCommand()"></label>
  <label>Internal TensorRT-LLM host<input id="host" value="0.0.0.0" oninput="updateCommand()"></label>
  <label>Host-published port<input id="port" type="number" value="8000" oninput="updateCommand()"></label>
  <label>Served model name<input id="served" oninput="updateCommand()"></label>
  <label>Max batch size<input id="mb" type="number" oninput="updateCommand()"></label>
  <label>Max sequence length<input id="ms" type="number" oninput="updateCommand()"></label>
  <label>Tensor parallel size<input id="tp" type="number" value="1" min="1" oninput="updateCommand()"></label>
  <label>KV cache dtype<input id="kv" oninput="updateCommand()"></label>
  <label>Free GPU memory fraction<input id="fg" type="number" step="0.01" oninput="updateCommand()"></label>
  <details><summary>Advanced</summary>
    <label>Pipeline parallel size<input id="pp" type="number" value="1" oninput="updateCommand()"></label>
    <label>Context parallel size<input id="cp" type="number" value="1" oninput="updateCommand()"></label>
    <label>MoE expert parallel size<input id="ep" type="number" value="1" oninput="updateCommand()"></label>
    <label>GPUs per node<input id="gpn" type="number" oninput="updateCommand()"></label>
  </details>
  <details><summary>Expert</summary>
    <label>Custom module dirs<input id="cmdirs" oninput="updateCommand()"></label>
    <label><input id="trc" type="checkbox" onchange="updateCommand()"> trust_remote_code</label>
    <label>Unsafe confirmation<input id="ack" placeholder="Type ENABLE UNSAFE" oninput="updateCommand()"></label>
    <label>Extra flags JSON<textarea id="extra" oninput="updateCommand()">{}</textarea></label>
  </details>
  <h3>Generated command</h3><pre id="command"></pre><button onclick="launch()">Deploy</button><div id="deploymsg"></div></div>`;
  if(state.models[0]){document.getElementById("model").value=state.models[0].name;document.getElementById("served").value=state.models[0].name;}
  updateCommand();
}
function configFromForm(){
  let extra={}; try{extra=JSON.parse(document.getElementById("extra").value||"{}")}catch{}
  const num=id=>{const v=document.getElementById(id).value;return v===""?null:Number(v)};
  return {model_name:document.getElementById("model").value,backend:document.getElementById("backend").value,host:document.getElementById("host").value,port:Number(document.getElementById("port").value),
    served_model_name:document.getElementById("served").value,max_batch_size:num("mb"),max_seq_len:num("ms"),tensor_parallel_size:num("tp")||1,
    pipeline_parallel_size:num("pp")||1,context_parallel_size:num("cp")||1,moe_expert_parallel_size:num("ep")||1,gpus_per_node:num("gpn"),
    kv_cache_dtype:document.getElementById("kv").value||null,free_gpu_memory_fraction:num("fg"),trust_remote_code:document.getElementById("trc").checked,
    custom_module_dirs:document.getElementById("cmdirs").value||null,unsafe_ack:document.getElementById("ack").value||null,extra_flags:extra};
}
function updateCommand(){
  const el=document.getElementById("command"); if(!el)return;
  const c=configFromForm(); const model=state.models.find(m=>m.name===c.model_name);
  if(!model){el.textContent="Select a model";return;}
  let a=["docker","run","-d","--name","trtllm-ui-PREVIEW","--gpus","all","--ipc=host","--ulimit","memlock=-1","--ulimit","stack=67108864","-p",`127.0.0.1:${c.port}:${c.port}`,"-v",`${model.host_path}:${model.container_path}:ro`,state.settings.docker_image,"trtllm-serve","serve",model.container_path,"--backend",c.backend,"--host",c.host,"--port",String(c.port),"--served_model_name",c.served_model_name||c.model_name];
  [["max_batch_size",c.max_batch_size],["max_seq_len",c.max_seq_len],["tensor_parallel_size",c.tensor_parallel_size!==1?c.tensor_parallel_size:null],["pipeline_parallel_size",c.pipeline_parallel_size!==1?c.pipeline_parallel_size:null],["context_parallel_size",c.context_parallel_size!==1?c.context_parallel_size:null],["moe_expert_parallel_size",c.moe_expert_parallel_size!==1?c.moe_expert_parallel_size:null],["gpus_per_node",c.gpus_per_node],["kv_cache_dtype",c.kv_cache_dtype],["free_gpu_memory_fraction",c.free_gpu_memory_fraction]].forEach(([k,v])=>{if(v!=null)a.push("--"+k,String(v));});
  if(c.trust_remote_code)a.push("--trust_remote_code"); if(c.custom_module_dirs)a.push("--custom_module_dirs",c.custom_module_dirs);
  Object.entries(c.extra_flags||{}).forEach(([k,v])=>{if(v===true)a.push(k.startsWith("--")?k:"--"+k);else if(v!==false&&v!=null&&v!=="")a.push(k.startsWith("--")?k:"--"+k,String(v));});
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
nav();show("dashboard");
