"""Start the Arize Phoenix dashboard with px.launch_app() and keep it running.

    python scripts/launch_phoenix.py      then open http://localhost:6006

Traces arrive over OTLP at http://<host>:6006/v1/traces (HTTP) or :4317 (gRPC).
Data is kept in PHOENIX_WORKING_DIR (a SQLite database), so it survives restarts.
"""
import os
import signal
import threading

import phoenix as px

host = os.getenv("PHOENIX_LAUNCH_HOST", "0.0.0.0")   # 0.0.0.0 so Docker can publish the port
port = int(os.getenv("PHOENIX_PORT", "6006"))

session = px.launch_app(host=host, port=port, use_temp_dir=False)
print(f"Phoenix is running: {session.url}  (Ctrl+C to stop)", flush=True)

stop = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: stop.set())
try:
    stop.wait()
except KeyboardInterrupt:
    pass
finally:
    px.close_app()
