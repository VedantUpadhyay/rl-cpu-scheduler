"""Shared project configuration — import this in all scripts."""
import os

# ---------------------------------------------------------------------------
# Scheduling window size — may be overridden at runtime via --n_processes
# ---------------------------------------------------------------------------
N_PROCESSES: int = 5   # default; scripts that accept --n_processes should use
                        # their parsed value instead of importing this directly.

# Base paths
PROJECT_ROOT = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-Personal"
    "/GRAD - FALL 23/UCSC/Capstone"
)
DATA_DIR = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-Personal"
    "/data/alibaba2018"
)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "docs/scheduler-research/results")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "docs/scheduler-research/scripts")

# Trace paths — always use these, never /tmp
TRACE_PATH = os.path.join(DATA_DIR, "trace_train_filtered.csv")
TEST_PATH  = os.path.join(DATA_DIR, "trace_test_filtered.csv")


def get_agent_dir(agent_name: str) -> str:
    """Get permanent results directory for an agent."""
    d = os.path.join(RESULTS_DIR, agent_name)
    os.makedirs(d, exist_ok=True)
    return d


def get_log_path(agent_name: str, filename: str) -> str:
    """Get permanent log file path for an agent."""
    return os.path.join(get_agent_dir(agent_name), filename)

# Never use /tmp — it is cleared on reboot
# Always use get_agent_dir() for new agents
