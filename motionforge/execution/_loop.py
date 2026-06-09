"""A private asyncio event loop running in a daemon thread.

Lets the synchronous Execution Adapter / fake RAPID server interface (SPEC §5.6) drive
asyncio sockets without forcing callers to be async. Submit coroutines with :meth:`run`.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine, Optional


class LoopThread:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="mf-asyncio", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro: Coroutine, timeout: Optional[float] = 10.0) -> Any:
        """Submit a coroutine to the loop thread and block for its result."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=2.0)
        if not self.loop.is_closed():
            self.loop.close()
