"""Live browser view of the showroom simulation: radar points, thermal camera images, true people,
tracks from radar / thermal / fused pipelines, and live counts.

    .venv/bin/python scripts/live_server.py            # then open http://localhost:8780

Over VS Code Remote-SSH the port is forwarded automatically (check the PORTS panel if not).
The server listens on 127.0.0.1 only.
"""
import argparse
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmw.live import LiveSim  # noqa: E402


class State:
    def __init__(self, cfg: dict, warmup: float):
        self.cfg = cfg
        self.warmup = warmup
        self.lock = threading.Lock()   # guards the simulation object
        self.cond = threading.Condition()
        self.ready = threading.Event()
        self.sim: LiveSim | None = None
        self.paused = False
        self.speed = 1.0
        self.reset_requested = False
        self.payload: bytes | None = None
        self.seq = 0

    def publish(self, data: bytes):
        with self.cond:
            self.payload = data
            self.seq += 1
            self.cond.notify_all()

    def wait(self, seq: int, timeout: float):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != seq, timeout=timeout)
            return self.seq, self.payload

    def controls(self) -> dict:
        s = self.sim
        return {"paused": self.paused, "speed": self.speed,
                "arrivals_per_min": round(s.crowd.arrival_rate * 60, 1), "static_mode": s.params.static_mode,
                "thermal_view": s.want_mosaic}


def sim_loop(st: State):
    # The simulation owns an OpenGL context, which belongs to the thread that creates it,
    # so creation, stepping and restarts all happen here.
    st.sim = LiveSim(st.cfg, warmup_s=st.warmup)
    st.ready.set()
    while True:
        if st.reset_requested:
            with st.lock:
                old = st.sim
                rate, mode, view = old.crowd.arrival_rate, old.params.static_mode, old.want_mosaic
                old.close()
                st.sim = LiveSim(st.cfg, seed=int(time.time()), warmup_s=st.warmup)
                st.sim.crowd.arrival_rate, st.sim.params.static_mode, st.sim.want_mosaic = rate, mode, view
                st.reset_requested = False
        if st.paused:
            time.sleep(0.05)
            continue
        t0 = time.perf_counter()
        with st.lock:
            frame = st.sim.step()
            dt, speed = st.sim.dt, st.speed
        st.publish(json.dumps(frame, separators=(",", ":")).encode())
        rest = (dt / speed if speed > 0 else 0.0) - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)


def make_handler(st: State, html_path: Path):
    cam_re = re.compile(r"^/camera/(\d+)\.png")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, html_path.read_bytes(), "text/html; charset=utf-8")
            st.ready.wait()
            if self.path == "/layout":
                with st.lock:
                    body = {**st.sim.layout_json(), "controls": st.controls()}
                return self._send(200, json.dumps(body).encode(), "application/json")
            m = cam_re.match(self.path)
            if m:
                with st.lock:
                    th = st.sim.thermal
                    k = int(m.group(1))
                    if th is None or k >= th.n:
                        return self._send(404, b"no such camera", "text/plain")
                    png = st.sim.camera_png(k)
                return self._send(200, png, "image/png")
            if self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                seq = -1
                try:
                    while True:
                        new, payload = st.wait(seq, 10.0)
                        if new == seq or payload is None:
                            self.wfile.write(b": keepalive\n\n")
                        else:  # a slow client simply skips to the newest frame
                            seq = new
                            self.wfile.write(b"data: " + payload + b"\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/control":
                return self._send(404, b"not found", "text/plain")
            st.ready.wait()
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, b"bad json", "text/plain")
            with st.lock:
                if "paused" in req:
                    st.paused = bool(req["paused"])
                if "speed" in req:
                    st.speed = max(0.0, float(req["speed"]))
                if req.get("reset"):
                    st.reset_requested = True
                if "arrivals_per_min" in req:
                    st.sim.crowd.arrival_rate = min(max(float(req["arrivals_per_min"]), 0.0), 120.0) / 60.0
                if req.get("static_mode") in ("scr", "minor_motion"):
                    st.sim.params.static_mode = req["static_mode"]
                if "thermal_view" in req:
                    st.sim.want_mosaic = bool(req["thermal_view"])
                body = st.controls()
            return self._send(200, json.dumps(body).encode(), "application/json")

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "showroom.yaml"))
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--warmup", type=float, default=240.0, help="seconds of crowd build-up before streaming")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    st = State(cfg, args.warmup)
    threading.Thread(target=sim_loop, args=(st,), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(st, ROOT / "web" / "live.html"))
    srv.daemon_threads = True
    print(f"Live viewer running: http://localhost:{args.port}  (warming up the crowd, first frames in a few s)",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
