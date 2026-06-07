"""Headless Isaac Sim smoke test: launch the Kit app, pump frames, close.

The real test on a new GPU (Blackwell sm_120) is whether the RTX renderer
initializes via EGL and the Kit extensions load — a bare import won't show that.
First run is slow (extension load / shader compile). Exits non-zero on failure.
"""
import sys, time
from isaacsim import SimulationApp

t0 = time.time()
app = SimulationApp({"headless": True})
print("SimulationApp (headless) up in %.1fs" % (time.time() - t0))

ok = False
try:
    for _ in range(60):
        app.update()                      # pump renderer + extension update loop
    ok = True
    print("app.update x60 OK")
except Exception as e:  # noqa: BLE001
    print("ISAAC SMOKE FAILED:", e, file=sys.stderr)
finally:
    app.close()

if not ok:
    sys.exit(1)
print("ISAAC_SMOKE_OK")
