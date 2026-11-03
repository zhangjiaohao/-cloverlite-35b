#!/usr/bin/env python3
"""Local Clover chat using Strata's resident engine protocol and design assets."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from clover_demo import ENGINE, MODEL, ROOT, tokenizer_from_gguf
from clover_metrics import collect_metrics

WEB = ROOT / 'web'
STRATA_WEB = ROOT / 'third_party/Strata/serve/web'
MAX_BODY = 128 * 1024
EOS = {248044, 248046}


class Backend:
    def __init__(self, model, engine, context, gpu, threads, fixed, sync, legacy, max_output_tokens):
        self.ready = False
        self.error = None
        self.busy = threading.Lock()
        self.write_lock = threading.Lock()
        self.tokenizer = None
        self.proc = None
        self.context = context
        self.max_output_tokens = max_output_tokens
        self.model = model
        self.engine = engine
        self.gpu = gpu
        self.threads = threads
        self.fixed = fixed
        self.sync = sync
        self.legacy = legacy
        threading.Thread(target=self.start, daemon=True).start()

    def start(self):
        try:
            self.tokenizer = tokenizer_from_gguf(self.model)
            args = [str(self.engine), '--model', str(self.model), '--experimental-clover-causal',
                    '--serve', '--max-context', str(self.context), '--chunk', '128',
                    '--expert-vram', str(self.gpu), '--cpu-threads', str(self.threads), '--no-warmup']
            env = os.environ.copy()
            # Reuse the block's hot experts across denoising rounds. Can be
            # disabled for comparison with STRATA_CLOVER_PREVIEW_REBALANCE=0.
            env.setdefault('STRATA_CLOVER_PREVIEW_REBALANCE', '1')
            env.setdefault('STRATA_CLOVER_SPEC_PROMOTE', '8')
            env.setdefault('STRATA_KQ256', '1')
            if self.fixed:
                env['STRATA_Q36_FIXED_SLOTS'] = '1'
            else:
                env.pop('STRATA_Q36_FIXED_SLOTS', None)
            if not self.legacy:
                env['STRATA_CLOVER_FAIR_SLOTS'] = '1'
            else:
                env.pop('STRATA_CLOVER_FAIR_SLOTS', None)
            env['CUDA_LAUNCH_BLOCKING'] = '1' if self.sync else '0'
            self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         text=True, bufsize=1, env=env)
            for line in self.proc.stdout:
                if line.startswith('READY '):
                    self.ready = True
                    return
                if line.startswith('ERR '):
                    raise RuntimeError(line.strip())
            raise RuntimeError(f'引擎启动失败（退出码 {self.proc.poll()}）')
        except Exception as exc:
            self.error = str(exc)

    def status(self):
        if self.ready and self.proc.poll() is not None:
            self.ready = False
            self.error = f'引擎已退出（{self.proc.returncode}）'
        return {'ready': self.ready, 'error': self.error, 'busy': self.busy.locked(),
                'model': 'Clover 35B A3B · Q4_K_M', 'modes': ['ar', 'diffusion', 'diffusion_spec'],
                'context': self.context, 'max_output_tokens': self.max_output_tokens, 'expert_vram_budget': self.gpu}

    def command(self, line):
        with self.write_lock:
            self.proc.stdin.write(line + '\n')
            self.proc.stdin.flush()

    def stop(self):
        if self.ready and self.busy.locked():
            self.command('STOP')

    def shutdown(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def prompt_ids(self, messages):
        if not isinstance(messages, list) or not messages or len(messages) > 100:
            raise ValueError('消息列表无效')
        parts = []
        for msg in messages:
            if not isinstance(msg, dict) or msg.get('role') not in ('user', 'assistant') or not isinstance(msg.get('content'), str):
                raise ValueError('消息格式无效')
            # Raw model-control markers in message content must not become special tokens.
            text = msg['content'].replace('<|', '＜|').replace('<think>', '＜think>')
            text = text.replace('</think>', '＜/think>')
            prefix = '<think>\n\n</think>\n\n' if msg['role'] == 'assistant' else ''
            parts.append(f"<|im_start|>{msg['role']}\n{prefix}{text}<|im_end|>\n")
        if messages[-1]['role'] != 'user':
            raise ValueError('最后一条消息必须来自用户')
        parts.append('<|im_start|>assistant\n<think>\n\n</think>\n\n')
        return self.tokenizer.encode(''.join(parts)).ids


class Handler(BaseHTTPRequestHandler):
    server_version = 'CloverLocal/0.1'
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        print('[web] ' + fmt % args, file=sys.stderr)

    def allowed_origin(self):
        origin = self.headers.get('Origin')
        if not origin:
            return True
        try:
            parsed = urlsplit(origin)
            if parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
                return False
            if parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost'):
                return parsed.port == self.server.server_port
            # Supported temporary tunnels forward the original Host to the local service.
            # Only accept the exact HTTPS origin matching that host, never arbitrary origins.
            host = self.headers.get('Host', '').lower()
            return (parsed.scheme == 'https' and parsed.hostname is not None and
                    parsed.hostname.endswith(('.trycloudflare.com', '.run.pinggy-free.link', '.free.pinggy.net')) and
                    parsed.port in (None, 443) and
                    host in (parsed.hostname, parsed.hostname + ':443'))
        except ValueError:
            return False

    def json_reply(self, code, payload):
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/api/status':
            return self.json_reply(200, self.server.backend.status())
        if path == '/api/metrics':
            return self.json_reply(200, collect_metrics())
        files = {'/': (WEB / 'index.html', 'text/html; charset=utf-8'),
                 '/app.js': (WEB / 'app.js', 'text/javascript; charset=utf-8'),
                 '/app.css': (WEB / 'app.css', 'text/css; charset=utf-8'),
                 '/strata/tokens.css': (STRATA_WEB / 'tokens.css', 'text/css; charset=utf-8'),
                 '/strata/components.css': (STRATA_WEB / 'components.css', 'text/css; charset=utf-8'),
                 '/strata/sprite.svg': (STRATA_WEB / 'sprite.svg', 'image/svg+xml')}
        item = files.get(path)
        if not item:
            self.send_error(404)
            return
        data = item[0].read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', item[1])
        self.send_header('Content-Length', str(len(data)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if not self.allowed_origin():
            return self.json_reply(403, {'error': '不允许跨站请求'})
        path = urlsplit(self.path).path
        if path == '/api/stop':
            self.server.backend.stop()
            return self.json_reply(200, {'ok': True})
        if path != '/api/chat':
            return self.json_reply(404, {'error': 'Not found'})
        backend = self.server.backend
        if not backend.status()['ready']:
            return self.json_reply(503, {'error': backend.error or '模型正在加载'})
        if not backend.busy.acquire(blocking=False):
            return self.json_reply(409, {'error': '模型正在回答其他请求'})
        started = False
        headers_sent = False
        disconnected = False

        def write_event(data):
            payload = (json.dumps(data, ensure_ascii=False) + '\n').encode()
            self.wfile.write(f'{len(payload):X}\r\n'.encode() + payload + b'\r\n')
            self.wfile.flush()
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if size <= 0 or size > MAX_BODY:
                raise ValueError('请求大小无效')
            body = json.loads(self.rfile.read(size))
            ids = backend.prompt_ids(body.get('messages'))
            mode = body.get('mode', 'ar')
            if mode not in ('ar', 'diffusion', 'diffusion_spec', 'dflash') or (mode == 'dflash' and not os.environ.get('STRATA_CLOVER_DFLASH_PACK')):
                raise ValueError('不支持的生成模式')
            requested = body.get('max_tokens', min(2048, backend.max_output_tokens))
            if isinstance(requested, bool) or not isinstance(requested, int):
                raise ValueError('生成长度必须是整数')
            max_tokens = requested
            if not 1 <= max_tokens <= backend.max_output_tokens:
                raise ValueError(f'生成长度需在 1 到 {backend.max_output_tokens} 之间')
            minimum = body.get('min_tokens', 0)
            if isinstance(minimum, bool) or not isinstance(minimum, int) or not 0 <= minimum <= max_tokens:
                raise ValueError('最少输出 tokens 必须是 0 到输出上限之间的整数')
            if mode == 'ar' and minimum:
                raise ValueError('最少输出设置目前只支持块扩散模式')
            threshold = body.get('diffusion_threshold', 0.95)
            if (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or
                    not math.isfinite(threshold) or not 0 <= threshold <= 1):
                raise ValueError('扩散置信度阈值必须在 0 到 1 之间')
            steps = body.get('diffusion_steps', 32)
            if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 32:
                raise ValueError('扩散轮数必须是 1 到 32 之间的整数')
            spec_accept = body.get('spec_accept', 'exact')
            if spec_accept not in ('exact', 'entropy'):
                raise ValueError('不支持的草稿验证方式')
            spec_delta = body.get('spec_delta', 0.6)
            if (isinstance(spec_delta, bool) or not isinstance(spec_delta, (int, float)) or
                    not math.isfinite(spec_delta) or not 0.01 <= spec_delta <= 1):
                raise ValueError('草稿接受参数必须在 0.01 到 1 之间')
            spec_width = body.get('spec_width', 4)
            if isinstance(spec_width, bool) or not isinstance(spec_width, int) or spec_width not in (2, 4, 8):
                raise ValueError('草稿窗口必须为 2、4 或 8')
            spec_refine = body.get('spec_refine', 0)
            if (isinstance(spec_refine, bool) or not isinstance(spec_refine, int) or
                    not 0 <= spec_refine < spec_width - 1):
                raise ValueError('草稿细化数量不合法')
            preview_tokens = (((max_tokens + 31) // 32) * 32 if mode == 'diffusion' else
                              max_tokens + spec_width - 1 if mode == 'diffusion_spec' else
                              max_tokens + 7 if mode == 'dflash' else max_tokens)
            if len(ids) + preview_tokens > backend.context:
                raise ValueError(f'对话超过 {backend.context} token 的上下文，请新建对话或减少生成长度')
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            headers_sent = True
            backend.command('GEN %d mode=%s threshold=%.6f min_new=%d diffusion_steps=%d spec_accept=%s spec_delta=%.6f spec_width=%d spec_refine=%d %s' %
                            (max_tokens, mode, threshold, minimum, steps, spec_accept, spec_delta, spec_width, spec_refine, ','.join(map(str, ids))))
            started = True
            tokens = []
            for line in backend.proc.stdout:
                if line.startswith('T '):
                    token = int(line.split()[1])
                    if token not in EOS:
                        tokens.append(token)
                        if not disconnected:
                            data = {'type': 'text', 'text': backend.tokenizer.decode(tokens, skip_special_tokens=True)}
                            try:
                                write_event(data)
                            except (BrokenPipeError, ConnectionResetError):
                                disconnected = True
                                backend.command('STOP')
                elif line.startswith('DONE '):
                    fields = line.split()
                    data = {'type': 'done', 'tokens': int(fields[1]), 'prompt_tokens': int(fields[2]),
                            'prefill_ms': float(fields[3]), 'decode_ms': float(fields[4]), 'finish': fields[5]}
                    if mode in ('diffusion_spec', 'dflash'):
                        data.update({'draft_accepted': int(fields[6]), 'draft_offered': int(fields[7])})
                    if mode in ('diffusion', 'diffusion_spec', 'dflash'):
                        for part in fields[8:]:
                            if part.startswith(('rounds=', 'blocks=', 'cycles=', 'non_greedy=')):
                                key, value = part.split('=', 1)
                                data[key] = int(value)
                    if not disconnected:
                        write_event(data)
                    return
                elif line.startswith('ERR '):
                    raise RuntimeError(line.strip())
            raise RuntimeError('模型引擎意外退出')
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            if not headers_sent:
                self.json_reply(400, {'error': str(exc)})
            elif not disconnected:
                write_event({'type': 'error', 'error': str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            if started:
                backend.stop()
        except Exception as exc:
            if not headers_sent:
                self.json_reply(500, {'error': str(exc)})
            elif not disconnected:
                try:
                    write_event({'type': 'error', 'error': str(exc)})
                except (BrokenPipeError, ConnectionResetError):
                    pass
        finally:
            if headers_sent and not disconnected:
                try:
                    self.wfile.write(b'0\r\n\r\n')
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            backend.busy.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', type=int, default=7860)
    ap.add_argument('--model', type=Path, default=MODEL)
    ap.add_argument('--engine', type=Path, default=ENGINE)
    ap.add_argument('--context', type=int, default=4096)
    ap.add_argument('--max-output-tokens', type=int, default=4096, help='单次请求最大输出 tokens，默认 4096')
    ap.add_argument('--expert-vram', type=float, default=7.5)
    ap.add_argument('--threads', type=int, default=10)
    ap.add_argument('--fixed-experts', action='store_true')
    ap.add_argument('--sync-cuda', action='store_true')
    ap.add_argument('--legacy-rebalance', action='store_true', help='use the original expert promotion order')
    args = ap.parse_args()
    if not 1024 <= args.context <= 32768 or not 0 <= args.expert_vram <= 8 or not 1 <= args.threads <= 12:
        ap.error('context 需为 1024–32768，expert-vram 为 0–8 GiB，threads 为 1–12')
    if not 1 <= args.max_output_tokens <= 32768:
        ap.error('max-output-tokens 需为 1–32768')
    backend = Backend(args.model, args.engine, args.context, args.expert_vram, args.threads, args.fixed_experts, args.sync_cuda, args.legacy_rebalance, args.max_output_tokens)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.backend = backend
    print(f'Clover UI: http://127.0.0.1:{args.port} （模型正在加载）', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        backend.shutdown()


if __name__ == '__main__':
    main()
