"""Compatibility shims loaded automatically by Python.

BasicSR 1.4.2 imports ``torchvision.transforms.functional_tensor``, which was
renamed to ``torchvision.transforms._functional_tensor`` in newer torchvision
releases. Keep the old module name available without patching site-packages.
"""

import sys

try:
    import torchvision.transforms._functional_tensor as _functional_tensor

    sys.modules.setdefault(
        "torchvision.transforms.functional_tensor",
        _functional_tensor,
    )
except Exception:
    pass
