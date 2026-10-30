import sys
from pathlib import Path

import jax

jax.config.update('jax_platform_name', 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
