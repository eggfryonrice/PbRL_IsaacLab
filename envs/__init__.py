import os
import importlib

current_dir = os.path.dirname(__file__)

submodules = [
    name
    for name in os.listdir(current_dir)
    if os.path.isdir(os.path.join(current_dir, name))
]

for module in submodules:
    try:
        globals()[module] = importlib.import_module(f".{module}", package=__name__)
    except ImportError as e:
        print(f"Could not import module {module}: {e}")
