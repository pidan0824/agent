"""Single-host Qwen worker. Only the worker owns weights and runs generate()."""

import asyncio
from concurrent.futures import Future
from contextlib import contextmanager
from math import isfinite
import multiprocessing as mp
from multiprocessing.connection import wait
from queue import Empty
from threading import Lock, Thread
from time import monotonic
from typing import NamedTuple
from uuid import uuid4


class QueuedRequest(NamedTuple):
    call_id: str
    request: dict
    deadline: float
    ticket: str


def _serve(factory, options, requests, replies, concurrency):
    try:
        backend = factory(**options)
        replies.send(backend.generator_id)
    except Exception as error:
        replies.send(error)
        return
    while True:
        first = requests.get()
        if first is None:
            return
        batch = [first]
        closing = False
        gather_until = monotonic() + 0.01
        # Start immediately when even the coalescing window would exhaust a deadline.
        while len(batch) < concurrency and batch[-1].deadline > gather_until:
            try:
                job = requests.get(timeout=max(0, gather_until - monotonic()))
            except Empty:
                break
            if job is None:
                closing = True
                break
            batch.append(job)
        active = [job for job in batch if monotonic() < job.deadline]
        try:
            if active:
                results = backend.complete_batch([(j.call_id, j.request, j.deadline) for j in active])
                if len(results) != len(active):
                    raise RuntimeError("Backend returned the wrong number of replies")
                for job, reply in zip(active, results):
                    replies.send((job.ticket, reply))
        except Exception as error:
            for job in active:
                replies.send((job.ticket, error))
        if closing:
            return


class LocalInferenceService:
    def __init__(self, factory, options, *, concurrency, timeout):
        if type(concurrency) is not int or concurrency < 1 or not isfinite(timeout) or timeout <= 0:
            raise ValueError("Expected positive concurrency and a finite positive request timeout")
        self.timeout = timeout
        self.pending = {}
        self.lock = Lock()
        self.stopped = False
        self.ctx = mp.get_context("spawn")
        self.requests = self.ctx.Queue()
        read, send = self.ctx.Pipe(duplex=False)
        self.read = read
        self.listener = None
        self.process = self.ctx.Process(target=_serve, args=(
            factory, options, self.requests, send, concurrency), daemon=True)
        self.process.start()
        send.close()
        try:
            if read not in wait([read, self.process.sentinel]):
                raise RuntimeError("Inference worker exited during startup")
            identity = read.recv()
            if isinstance(identity, Exception):
                raise identity
            self.generator_id = identity
            self.listener = Thread(target=self._receive, daemon=True)
            self.listener.start()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        try:
            while True:
                ticket, value = self.read.recv()
                with self.lock:
                    future = self.pending.get(ticket)
                    if future is not None and not future.done():
                        future.set_result(value)
        except (EOFError, OSError):
            with self.lock:
                self.stopped = True
                for future in self.pending.values():
                    if not future.done():
                        future.set_result(RuntimeError("Inference worker exited during request"))

    @contextmanager
    def _submitted(self, request, call_id, deadline):
        future = Future()
        ticket = uuid4().hex
        with self.lock:
            if self.stopped:
                raise RuntimeError("Inference service closed")
            self.pending[ticket] = future
        try:
            self.requests.put(QueuedRequest(call_id, request, deadline, ticket))
            try:
                yield future
            except TimeoutError as error:
                raise TimeoutError(f"Inference deadline exceeded: {call_id}") from error
            if monotonic() >= deadline:
                raise TimeoutError(f"Inference deadline exceeded: {call_id}")
        finally:
            with self.lock:
                self.pending.pop(ticket)

    def complete(self, request, *, call_id):
        deadline = monotonic() + self.timeout
        with self._submitted(request, call_id, deadline) as future:
            reply = future.result(timeout=max(0, deadline - monotonic()))
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def acomplete(self, request, *, call_id):
        deadline = monotonic() + self.timeout
        with self._submitted(request, call_id, deadline) as future:
            reply = await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)),
                                           timeout=max(0, deadline - monotonic()))
        if isinstance(reply, Exception):
            raise reply
        return reply

    def close(self):
        proc = self.process
        if proc.is_alive():
            self.requests.put(None)
            proc.join(1)
        if proc.is_alive():
            proc.terminate()
            proc.join(1)
        if proc.is_alive():
            proc.kill()
            proc.join()
        if self.listener is not None:
            self.listener.join()
        self.read.close()
        self.requests.cancel_join_thread()
        self.requests.close()
        proc.close()
