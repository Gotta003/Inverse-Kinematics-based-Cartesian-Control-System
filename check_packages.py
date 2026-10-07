import sys

checks = {
    "numpy": lambda: __import__("numpy").__version__,
    "scipy": lambda: __import__("scipy").__version__,
    "matplotlib": lambda: __import__("matplotlib").__version__,
    "mujoco": lambda: __import__("mujoco").__version__,
    "lerobot": lambda: getattr(__import__("lerobot"), "__version__", "installed"),
    "pytest": lambda: __import__("pytest").__version__,
}

GREEN = "\033[0;32m"
RED = "\033[0;31m"
RESET = "\033[0m"
failed = []

for pkg, ver_func in checks.items():
    try:
        ver = ver_func()
        print(f"{GREEN}{pkg:<20} {ver}{RESET}")
    except Exception:
        print(f"{RED}{pkg:<20} not found{RESET}")
        failed.append(pkg)

if failed:
    print(f"\n{RED}The following packages are missing: {', '.join(failed)}{RESET}")
    sys.exit(1)
else:
    print(f"\n{GREEN}All packages are installed!{RESET}")