# minflux_msr/__init__.py
from .utils import slug
from .state import (
    mfx, mbm,                # first dataset (back-compat)
    mfx_map, mbm_map,        # NEW: all datasets
    reset as reset_state,
    set_mfx_for, set_mbm_for # NEW: setters per dataset
)
from .io import (
    collect_zarr_fields,
    parse_msr_to_tree,        # old (kept)
    parse_msr_general,        # NEW: modern+legacy, fills state.mfx/mbm
    pick_one_msr,
)
from .export import (
    export_arrays,            # NEW: export using arrays (mfx/mbm)
    export_selected_fields,   # existing, unchanged
    make_export_base,         # existing
)
