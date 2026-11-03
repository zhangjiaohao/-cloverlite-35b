"""Read GPU and host/container memory metrics without extra runtime dependencies."""
from pathlib import Path
import subprocess


def _read_int(path):
    try:
        text = Path(path).read_text().strip()
        if text == 'max':
            return None
        value = int(text)
        return value if 0 <= value < (1 << 60) else None
    except (OSError, ValueError):
        return None


def collect_metrics():
    result = {'gpu': None, 'host_memory': None, 'container_memory': None}
    try:
        p = subprocess.run(['nvidia-smi', '--id=0',
                            '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True,
                           timeout=2, check=True)
        name, util, used, total, temp = [x.strip() for x in p.stdout.strip().splitlines()[0].rsplit(',', 4)]
        result['gpu'] = {'name': name, 'utilization_percent': float(util),
                         'memory_used_bytes': int(used) * 1024**2,
                         'memory_total_bytes': int(total) * 1024**2, 'temperature_c': float(temp)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    try:
        data = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemAvailable'):
                data[key] = int(value.strip().split()[0]) * 1024
        if 'MemTotal' in data and 'MemAvailable' in data:
            result['host_memory'] = {'total_bytes': data['MemTotal'], 'available_bytes': data['MemAvailable'],
                                     'used_bytes': data['MemTotal'] - data['MemAvailable']}
    except (OSError, ValueError, IndexError):
        pass
    # The cgroup mount exposes this container's root on this machine. On cgroup v2 use memory.max/current.
    for limit_path, used_path in [('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
                                  ('/sys/fs/cgroup/memory/memory.limit_in_bytes',
                                   '/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        if Path(used_path).exists():
            limit, used = _read_int(limit_path), _read_int(used_path)
            if used is not None:
                result['container_memory'] = {'used_bytes': used, 'limit_bytes': limit}
            break
    return result
