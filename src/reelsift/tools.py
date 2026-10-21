"""Locate and run the FFmpeg command-line tools, and other long-running helper processes."""

import functools
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path


class ToolNotFoundError(RuntimeError):
    pass


class Cancelled(BaseException):  # like KeyboardInterrupt: not an error
    """Work was cancelled. Not an Exception, so nothing treats it as a failure to retry or to
    fall back from (another encoder, the next file): it goes straight up to whoever cancelled."""


@functools.cache
def tool(name: str) -> str:
    """Return the path of ffmpeg or ffprobe.

    Looked up in this order: REELSIFT_FFMPEG_DIR, PATH, then the winget package folder, whose
    alias only appears on PATH in shells started after installation.
    """
    env_dir = os.environ.get("REELSIFT_FFMPEG_DIR")
    if env_dir:
        for candidate in (Path(env_dir) / name, Path(env_dir) / f"{name}.exe"):
            if candidate.is_file():
                return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    local = os.environ.get("LOCALAPPDATA")
    if local:
        for candidate in sorted(
            (Path(local) / "Microsoft/WinGet/Packages").glob(f"Gyan.FFmpeg*/*/bin/{name}.exe")
        ):
            return str(candidate)
    raise ToolNotFoundError(
        f"{name} was not found. Install FFmpeg (on Windows: winget install Gyan.FFmpeg) "
        "or set REELSIFT_FFMPEG_DIR to the folder that contains it."
    )


class _WindowsJob:
    """A Windows job object that a process and everything it starts belong to: killing the job
    kills them all, even after the first process has exited."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        for name in ("SetInformationJobObject", "QueryInformationJobObject"):
            getattr(k32, name).argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID,
                                           wintypes.DWORD] + ([ctypes.c_void_p] if name.startswith(
                                               "Query") else [])  # fmt: skip
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                        ("PerJobUserTimeLimit", ctypes.c_longlong), ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]  # fmt: skip

        class Extended(ctypes.Structure):
            _fields_ = [("Basic", Basic), ("Io", ctypes.c_ulonglong * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]  # fmt: skip

        class Accounting(ctypes.Structure):
            _fields_ = [("Times", ctypes.c_longlong * 4), ("TotalPageFaultCount", wintypes.DWORD),
                        ("TotalProcesses", wintypes.DWORD), ("ActiveProcesses", wintypes.DWORD),
                        ("TotalTerminatedProcesses", wintypes.DWORD)]  # fmt: skip

        self.ctypes, self.k32, self.Accounting = ctypes, k32, Accounting
        self.handle = k32.CreateJobObjectW(None, None)
        info = Extended()
        info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        k32.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info))

    def add(self, proc: subprocess.Popen) -> None:
        self.k32.AssignProcessToJobObject(self.handle, int(proc._handle))

    def active(self) -> int:
        info = self.Accounting()
        ok = self.k32.QueryInformationJobObject(self.handle, 1, self.ctypes.byref(info),
                                                self.ctypes.sizeof(info), None)  # fmt: skip
        return info.ActiveProcesses if ok else 0

    def kill(self, wait_s: float = 10.0) -> None:
        """Kill every process in the job and wait until they have gone."""
        self.k32.TerminateJobObject(self.handle, 1)
        deadline = time.monotonic() + wait_s
        while self.active() and time.monotonic() < deadline:
            time.sleep(0.05)

    def close(self) -> None:
        if self.handle:
            self.k32.CloseHandle(self.handle)  # kills what is left (kill on close)
            self.handle = None


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill a process and everything it started, even after the first process has exited
    (processes started by stream() are in a job object on Windows, a process group elsewhere)."""
    job = getattr(proc, "reelsift_job", None)
    if job is not None:
        job.kill()
    elif os.name != "nt" and getattr(proc, "reelsift_group", False):
        import signal

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:  # until no process of the group is left
            try:
                os.killpg(proc.pid, 0)
            except (ProcessLookupError, PermissionError):
                break
            time.sleep(0.05)
    elif proc.poll() is None and os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    if proc.poll() is None:
        proc.kill()


def stop_all() -> None:
    """Stop every process started through run() or stream() that is still running."""
    with _running_lock:
        running = list(_running)
    for proc in running:
        kill_tree(proc)


_local = threading.local()


@contextmanager
def cancellable(event: threading.Event) -> Iterator[None]:
    """Run tools in this thread under a cancel event: once cancel(event) is called, running
    processes are killed, no new ones start, and run() raises Cancelled."""
    _local.cancel = event
    try:
        yield
    finally:
        _local.cancel = None


def cancel(event: threading.Event) -> None:
    """Cancel the work running under this event and kill its processes."""
    with _running_lock:
        event.set()
        running = [p for p, e in _running.items() if e is event]
    for proc in running:
        kill_tree(proc)


def is_cancelled() -> bool:
    """Whether the work in this thread was cancelled."""
    event = getattr(_local, "cancel", None)
    return event is not None and event.is_set()


def check_cancelled() -> None:
    """Raise Cancelled if the work in this thread was cancelled."""
    event = getattr(_local, "cancel", None)
    if event is not None and event.is_set():
        raise Cancelled()


@functools.cache
def encoder_works(name: str) -> bool:
    """Whether FFmpeg can actually encode with this encoder here (NVENC needs a capable GPU)."""
    try:
        run(
            [
                tool("ffmpeg"),
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=size=256x256:duration=0.2",
                "-c:v",
                name,
                "-f",
                "null",
                "-",
            ]
        )
    except (RuntimeError, ToolNotFoundError):
        return False
    return True


@functools.cache
def filter_available(name: str) -> bool:
    try:
        out = run([tool("ffmpeg"), "-v", "error", "-filters"], text=True).stdout
    except (RuntimeError, ToolNotFoundError):
        return False
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


def h264_args(quality: int = 23, cpu: bool = False) -> list[str]:
    """Encoder arguments for H.264, using NVENC when it works (and cpu is False), x264 otherwise."""
    if not cpu and encoder_works("h264_nvenc"):
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(quality)]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(quality)]


_running: dict[subprocess.Popen, threading.Event | None] = {}
_running_lock = threading.Lock()


def run(args: list[str], *, text: bool = False) -> subprocess.CompletedProcess:
    """Run an FFmpeg tool, raising with its error output on failure. While it runs, the process
    is known to stop_all() and, inside cancellable(), to cancel()."""
    event = getattr(_local, "cancel", None)
    with _running_lock:
        # Checked and started under the lock, so cancel() either stops this launch or kills it.
        if event is not None and event.is_set():
            raise Cancelled()
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text)
        _running[proc] = event
    try:
        stdout, stderr = proc.communicate()
    finally:
        with _running_lock:
            _running.pop(proc, None)
    check_cancelled()
    result = subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)
    if result.returncode != 0:
        stderr = result.stderr if text else result.stderr.decode(errors="replace")
        raise RuntimeError(f"{Path(args[0]).name} failed: {stderr.strip()[-2000:]}")
    return result


def stream(
    args: list[str],
    request: str,
    on_line: Callable[[str], None],
    log: Path,
    *,
    env: dict | None = None,
    cwd: Path | None = None,
    stall_s: float = 900.0,
) -> int:
    """Run a helper process that answers with lines on standard output: send it `request` on
    standard input, call on_line for each line as it arrives, and append its standard error to
    `log`. Cancellable like run() (the whole process tree is killed, and Cancelled raised once it
    has exited). A process silent for stall_s seconds is killed. Returns the exit code."""
    import queue

    event = getattr(_local, "cancel", None)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as err, _running_lock:
        if event is not None and event.is_set():
            raise Cancelled()
        proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, env=env, cwd=cwd,
            start_new_session=os.name != "nt",
        )  # fmt: skip
        if os.name == "nt":
            # Assigned before the helper gets far enough to start anything (it is Python,
            # still starting up), so everything it starts is in the job too.
            proc.reelsift_job = _WindowsJob()
            proc.reelsift_job.add(proc)
        else:
            proc.reelsift_group = True
        _running[proc] = event
    lines: queue.Queue = queue.Queue()

    def read() -> None:
        for raw in proc.stdout:
            lines.put(raw.decode("utf-8", "replace").rstrip("\r\n"))
        lines.put(None)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        try:
            proc.stdin.write(request.encode("utf-8"))
            proc.stdin.close()
        except OSError:
            pass  # it exited early; its output says why
        quiet_since = time.monotonic()
        while True:
            try:
                line = lines.get(timeout=0.5)
            except queue.Empty:
                if event is not None and event.is_set():
                    kill_tree(proc)
                elif time.monotonic() - quiet_since > stall_s:
                    kill_tree(proc)
                    raise RuntimeError(
                        f"{Path(args[0]).name} stopped responding; see {log}"
                    ) from None
                continue
            if line is None:
                break
            quiet_since = time.monotonic()
            if line:
                on_line(line)
        # Its output has ended; it should exit now. The stall limit still applies.
        closed = time.monotonic()
        while proc.poll() is None:
            if event is not None and event.is_set():
                kill_tree(proc)
            elif time.monotonic() - closed > stall_s:
                kill_tree(proc)
                raise RuntimeError(f"{Path(args[0]).name} didn't exit after finishing; see {log}")
            time.sleep(0.1)
        code = proc.wait()
    finally:
        kill_tree(proc)  # anything it left behind
        proc.wait()
        job = getattr(proc, "reelsift_job", None)
        if job is not None:
            job.close()
        with _running_lock:
            _running.pop(proc, None)
    check_cancelled()
    return code
