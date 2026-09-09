# Local crash diagnostics

Restart the backend after installing this change. `logs/backend-crash.log`
(or `$LOG_DIR/backend-crash.log`) is append-only and independent of the terminal,
ordinary logging queues, and `backend.log` rotation. It records a UTC startup
marker and PID, uncaught main/thread exceptions, unraisable exceptions, and
Python faulthandler output for supported fatal signals including SIGSEGV,
SIGABRT, SIGBUS, SIGILL and SIGFPE. Fatal dumps include all Python thread stacks
available to the interpreter. Native C-stack availability depends on Python
version/build; this is not an OS core dump or a capture of all process memory.

`scripts/start.sh` sends backend stdout directly to its per-run `start-*.log`,
also sends backend stderr directly to this file, and enables
Python fault reporting from interpreter startup. Consequently it contains
ordinary stderr logging too. Normal task exception tracebacks continue to be
stored in `logs/backend.log`; task summaries remain in
`outputs/<task-id>/logs/pipeline.log`. Child-process coverage depends on the
child's own diagnostics and the pipeline's stdout/stderr capture; remote
services require their own host logs.

Do not rotate, truncate, or replace backend-crash.log while the backend is
running: the fatal handler retains its descriptor. Archive it only while the
service is stopped. SIGKILL, power loss, disk failure/full disk, and severe
runtime corruption cannot be guaranteed to produce a final stack. Python
faulthandler itself limits frames/threads and string lengths for signal safety;
"all threads" is not a guarantee of unlimited diagnostic content.

Validation uses isolated child processes with core dumps disabled and terminal
output discarded. SIGABRT/SIGSEGV/SIGBUS must still leave current and background
thread stacks in the file. No production process is intentionally crashed.
