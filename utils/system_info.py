# nav_mpc/utils/system_info.py

import platform
import os

def print_system_info():
    """
    Print basic system info so timing results can be interpreted later.
    Works on Linux; uses /proc/cpuinfo if available.
    """
    uname = platform.uname()
    print("=========================== System info ===========================")
    print(f"OS:       {uname.system} {uname.release}")
    print(f"Machine:  {uname.machine}")

    # CPU model name (Linux)
    cpu_model = "Unknown"
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    cpu_model = line.strip().split(": ", 1)[1]
                    break
    print(f"CPU:      {cpu_model}")

    # Core count (no external deps)
    logical = os.cpu_count()
    print(f"Cores:    {logical} logical")

    print("===================================================================")
