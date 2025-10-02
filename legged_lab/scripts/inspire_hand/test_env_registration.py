# Minimal registration check for inspirehand_grasp
from isaaclab.app import AppLauncher

# 1) Boot Omniverse kernel BEFORE importing envs (avoids 'carb' errors)
app = AppLauncher(headless=True)
simulation_app = app.app

# 2) Import envs → runs legged_lab/envs/__init__.py (does the registrations)
import legged_lab.envs as _envs  # noqa: F401

# 3) Inspect the registry
from legged_lab.utils.task_registry import task_registry

print("[INFO] Registered task names      :", list(task_registry.task_classes.keys()))
print("[INFO] Registered env cfg entries :", list(task_registry.env_cfgs.keys()))
print("[INFO] Registered agent cfg entries:", list(task_registry.train_cfgs.keys()))

# 4) Try to build the env
TASK = "inspirehand_grasp"
try:
    env_cfg, agent_cfg = task_registry.get_cfgs(TASK)
    env_class = task_registry.get_task_class(TASK)
    env = env_class(env_cfg, render_mode=None)
    print(f"[SUCCESS] Built '{TASK}' env: {env}")
    env.close()
except KeyError as ke:
    print(f"[ERROR] Task '{TASK}' not found in registry. Available: {list(task_registry.task_classes.keys())}")
except Exception as e:
    print(f"[ERROR] Could not create '{TASK}' env:", e)

# 5) Clean shutdown
simulation_app.close()
