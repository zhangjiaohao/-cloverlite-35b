'use strict';
const $ = id => document.getElementById(id);
const state = {messages: [], busy: false, ready: false, view: 'chat'};
const storageKey = 'clover.local.conversation.v1';
try { state.messages = JSON.parse(localStorage.getItem(storageKey) || '[]'); if (!Array.isArray(state.messages)) state.messages = []; } catch (_) { state.messages = []; }
function save() { try { localStorage.setItem(storageKey, JSON.stringify(state.messages)); } catch (_) {} }
function render() {
  const box = $('conversation'); box.replaceChildren();
  $('welcome').hidden = state.messages.length > 0;
  const first = state.messages.find(m => m.role === 'user'); $('session-name').textContent = first ? first.content.slice(0, 24) : '新对话';
  for (const msg of state.messages) {
    const row = document.createElement('article'); row.className = 'message ' + msg.role;
    const avatar = document.createElement('div'); avatar.className = 'avatar'; avatar.textContent = msg.role === 'user' ? '你' : '✳';
    const body = document.createElement('div'); body.className = 'message-body';
    const label = document.createElement('div'); label.className = 'message-label'; label.textContent = msg.role === 'user' ? '你' : 'Clover';
    const bubble = document.createElement('div'); bubble.className = 'bubble'; bubble.textContent = msg.content || (state.busy ? '正在思考…' : '');
    body.append(label, bubble); row.append(avatar, body); box.append(row);
  }
  $('viewport').scrollTop = $('viewport').scrollHeight;
}
function setBusy(value) { state.busy = value; $('send').disabled = value || !state.ready; $('stop').hidden = !value; $('input').disabled = value; $('generation-mode').disabled = value; $('diffusion-steps').disabled = value; $('spec-accept').disabled = value; $('min-tokens').disabled = value; }
async function refreshStatus() {
  try {
    const data = await (await fetch('/api/status', {cache:'no-store'})).json();
    state.ready = data.ready; $('status').className = 'status ' + (data.ready ? 'ready' : data.error ? 'error' : '');
    $('status').lastElementChild.textContent = data.error ? '引擎异常' : data.ready ? '模型已就绪' : '模型加载中';
    $('status').title = data.error || data.model;
    if (Number.isInteger(data.max_output_tokens) && Number.isInteger(data.context)) {
      const max = Math.min(data.max_output_tokens, data.context - 1);
      $('max-tokens').max = String(max);
      $('max-tokens').title = `服务上限 ${data.max_output_tokens} tokens，上下文 ${data.context} tokens；对话也占上下文`;
      if (+$('max-tokens').value > max) $('max-tokens').value = String(max);
    }
    $('send').disabled = state.busy || !data.ready;
  } catch (_) { state.ready = false; $('status').className = 'status error'; $('status').lastElementChild.textContent = '连接已断开'; $('send').disabled = true; }
}
function addMessage(role, content) { state.messages.push({role, content}); save(); render(); }
async function submit(text) {
  if (!text.trim() || state.busy || !state.ready) return;
  const limit = $('max-tokens');
  if (!Number.isInteger(limit.valueAsNumber) || limit.valueAsNumber < 1 || limit.valueAsNumber > +limit.max) {
    limit.setCustomValidity(`请输入 1 到 ${limit.max} 的整数`); limit.reportValidity(); return;
  }
  limit.setCustomValidity('');
  const diffusion = $('generation-mode').value !== 'ar';
  const minimum = diffusion ? $('min-tokens').valueAsNumber : 0;
  if (!Number.isInteger(minimum) || minimum < 0 || minimum > limit.valueAsNumber) {
    $('min-tokens').setCustomValidity('最少输出必须是 0 到输出上限之间的整数'); $('min-tokens').reportValidity(); return;
  }
  $('min-tokens').setCustomValidity('');
  addMessage('user', text.trim()); $('input').value = ''; $('input').style.height = 'auto';
  const history = state.messages.map(({role,content}) => ({role,content}));
  addMessage('assistant', ''); setBusy(true); render();
  try {
    const r = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({messages:history,max_tokens:limit.valueAsNumber,min_tokens:minimum,diffusion_steps:+$('diffusion-steps').value,spec_accept:$('spec-accept').value,mode:$('generation-mode').value})});
    if (!r.ok) { const body = await r.json(); throw new Error(body.error || `HTTP ${r.status}`); }
    const reader = r.body.getReader(), decoder = new TextDecoder(); let pending = '';
    while (true) {
      const {value,done} = await reader.read(); pending += decoder.decode(value || new Uint8Array(), {stream:!done});
      const lines = pending.split('\n'); pending = lines.pop();
      for (const line of lines) {
        if (!line) continue; const event = JSON.parse(line);
        if (event.type === 'text') { state.messages.at(-1).content = event.text; render(); }
        else if (event.type === 'done') {
          if (event.finish === 'error') throw new Error('引擎生成失败，请重试');
          const speed = event.decode_ms > 0 ? Math.max(0,event.tokens - ($('generation-mode').value === 'ar' ? 1 : 0)) * 1000 / event.decode_ms : 0;
          const reason = {length:'达到输出上限', stop:'模型正常结束', cancel:'已停止'}[event.finish] || event.finish;
          $('perf').textContent = `${speed ? speed.toFixed(1) + ' tokens/s · ' : ''}${reason} · ${event.tokens} tokens`;
          $('perf').title = event.finish === 'length' ? '若内容未完成，请提高输出上限并重新生成，或继续对话。' : '';
        } else if (event.type === 'error') throw new Error(event.error);
      }
      if (done) break;
    }
    if (!state.messages.at(-1).content) state.messages.at(-1).content = '（没有生成文本）';
  } catch (e) { state.messages.at(-1).content = '生成失败：' + e.message; }
  finally { save(); setBusy(false); render(); $('input').focus(); refreshStatus(); }
}
function showView(view) {
  state.view = view; $('chat-view').hidden = view !== 'chat'; $('monitor-view').hidden = view !== 'monitor';
  $('page-title').textContent = view === 'chat' ? '对话' : '资源监控';
  $('nav-chat').classList.toggle('active', view === 'chat'); $('nav-monitor').classList.toggle('active', view === 'monitor');
  $('mobile-monitor').textContent = view === 'chat' ? '▥' : '◉';
  $('mobile-monitor').title = view === 'chat' ? '资源监控' : '返回对话';
  $('mobile-monitor').setAttribute('aria-label', $('mobile-monitor').title);
  if (view === 'monitor') refreshMetrics();
}
const gib = value => (value / 1073741824).toFixed(2);
function paintMeter(id, used, total) {
  const valid = Number.isFinite(used) && Number.isFinite(total) && total > 0;
  const pct = valid ? Math.min(100, Math.max(0, used * 100 / total)) : 0;
  $(id + '-bar').style.width = `${pct}%`;
  $(id + '-percent').textContent = valid ? `${pct.toFixed(1)}%` : '—';
}
async function refreshMetrics() {
  if (state.view !== 'monitor') return;
  try {
    const r = await fetch('/api/metrics', {cache: 'no-store'}); if (!r.ok) throw new Error('metrics');
    const data = await r.json(), gpu = data.gpu, container = data.container_memory, host = data.host_memory;
    $('gpu-memory').textContent = gpu ? `${gib(gpu.memory_used_bytes)} / ${gib(gpu.memory_total_bytes)} GiB` : '暂不可用';
    $('gpu-name').textContent = gpu ? gpu.name : '无法读取 NVIDIA 显卡';
    paintMeter('gpu-memory', gpu?.memory_used_bytes, gpu?.memory_total_bytes);
    $('gpu-util').textContent = gpu ? `${gpu.utilization_percent.toFixed(0)}%` : '暂不可用';
    $('gpu-temp').textContent = gpu ? `温度 ${gpu.temperature_c.toFixed(0)} °C` : '无法读取 GPU 利用率';
    paintMeter('gpu-util', gpu?.utilization_percent, 100);
    $('container-memory').textContent = container ? (container.limit_bytes ? `${gib(container.used_bytes)} / ${gib(container.limit_bytes)} GiB` : `${gib(container.used_bytes)} GiB / 未限制`) : '暂不可用';
    paintMeter('container-memory', container?.used_bytes, container?.limit_bytes);
    $('host-memory').textContent = host ? `${gib(host.used_bytes)} / ${gib(host.total_bytes)} GiB` : '暂不可用';
    $('host-memory-available').textContent = host ? `宿主机可用 ${gib(host.available_bytes)} GiB` : '无法读取宿主机内存';
    paintMeter('host-memory', host?.used_bytes, host?.total_bytes);
    $('metrics-updated').textContent = `更新于 ${new Date().toLocaleTimeString()} · 每 3 秒刷新`;
  } catch (_) { $('metrics-updated').textContent = '暂时无法读取资源数据，正在重试'; }
}
$('nav-chat').addEventListener('click', () => showView('chat'));
$('nav-monitor').addEventListener('click', () => showView('monitor'));
$('mobile-monitor').addEventListener('click', () => showView(state.view === 'chat' ? 'monitor' : 'chat'));
$('generation-mode').addEventListener('change', e => {
  const diffusion = e.target.value === 'diffusion', spec = e.target.value === 'diffusion_spec';
  $('threshold-label').hidden = !diffusion; $('spec-label').hidden = !spec; $('minimum-label').hidden = !(diffusion || spec);
  $('mode-description').textContent = diffusion ? '块扩散：完整生成一块后输出。' : spec ? '扩散草稿 + 因果验证：熵阈值可能接受非贪心草稿，结果与纯扩散不同。' : '因果生成：逐 token 输出。';
  $('perf').textContent = '';
});
$('max-tokens').addEventListener('input', e => e.target.setCustomValidity(''));
$('min-tokens').addEventListener('input', e => e.target.setCustomValidity(''));
setInterval(refreshMetrics, 3000);
$('form').addEventListener('submit' , e => { e.preventDefault(); submit($('input').value); });
$('input').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $('form').requestSubmit(); } });
$('input').addEventListener('input', e => { e.target.style.height = 'auto'; e.target.style.height = Math.min(e.target.scrollHeight, 180) + 'px'; });
$('stop').addEventListener('click', () => fetch('/api/stop', {method:'POST'}));
$('new-chat').addEventListener('click', () => { if (state.busy) return; showView('chat'); state.messages = []; $('perf').textContent = ''; save(); render(); $('input').focus(); });
for (const el of document.querySelectorAll('[data-prompt]')) el.addEventListener('click', () => submit(el.dataset.prompt));
$('mobile-new').addEventListener('click', () => $('new-chat').click());
$('theme').addEventListener('click', () => { document.documentElement.classList.toggle('dark'); localStorage.setItem('clover.theme', document.documentElement.classList.contains('dark') ? 'dark' : 'light'); });
if (localStorage.getItem('clover.theme') === 'dark') document.documentElement.classList.add('dark');
document.addEventListener('keydown', e => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); $('new-chat').click(); } });
render(); refreshStatus(); setInterval(refreshStatus, 3000);
