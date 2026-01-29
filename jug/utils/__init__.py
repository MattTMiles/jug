"""JUG utilities module.

IMPORTANT: This module is designed to NOT import JAX at import time.
JAX-dependent utilities (like device management) should be imported
explicitly from their submodules:

    from jug.utils.device import get_device, set_device_preference
    from jug.utils.jax_setup import ensure_jax_x64

Non-JAX utilities are always available:

    from jug.utils.constants import SECS_PER_DAY, K_DM_SEC
"""

# Only export non-JAX utilities at module level
from jug.utils.constants import *

# JAX-dependent utilities are NOT imported here to avoid eager JAX import.
# Users should import them explicitly:
#   from jug.utils.device import get_device
#   from jug.utils.jax_setup import ensure_jax_x64
#   from jug.utils.jax_cache import configure_jax_compilation_cache
