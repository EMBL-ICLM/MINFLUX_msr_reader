# minflux_msr/io.py
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import zarr
import specpy

from .utils import slug
from .state import reset as reset_state, set_mfx_for, set_mbm_for

# -------- existing: collect_zarr_fields (unchanged) --------
def collect_zarr_fields(zroot: str) -> List[Dict[str, Any]]:
    g = zarr.open(zroot, mode="r")
    out: List[Dict[str, Any]] = []
    def visitor(path, obj):
        if path == "":
            return
        is_array = hasattr(obj, "shape") and hasattr(obj, "dtype")
        kind = "array" if is_array else "group"
        ent: Dict[str, Any] = {"path": path, "kind": kind}
        if is_array:
            ent["shape"] = tuple(getattr(obj, "shape", ()))
            ent["length"] = int(obj.shape[0]) if obj.shape else 0
            dt = getattr(obj, "dtype", "")
            ent["dtype"] = str(dt)

            # NEW: recursively describe structured dtype fields (nested supported)
            try:
                tree = _dtype_fields_tree(dt, ent["length"])
                if tree:
                    ent["dtype_fields"] = tree
                    # optional: expose the singleton root name for UI convenience
                    if len(tree) == 1 and tree[0].get("kind") == "struct-root":
                        ent["dtype_singleton_root"] = tree[0]["name"]
            except Exception:
                pass
            
            names = getattr(dt, "names", None)
            if names:
                sub = []
                for name in names:
                    field_dt = dt.fields[name][0]  # numpy dtype of the field
                    # Try to get the per-element subshape, e.g. (3,) for loc
                    subshape = getattr(field_dt, "shape", ())
                    # Base dtype string (no subshape) for display
                    fdt_str = str(getattr(field_dt, "base", field_dt))
                    sub.append({
                        "name": name,
                        "dtype": fdt_str,
                        "shape": tuple(subshape) if subshape else (),
                        "logical_shape": _logical_shape_str_for_field(ent["length"], tuple(subshape) if subshape else ()),
                    })
                ent["dtype_fields"] = sub
        out.append(ent)
    g.visititems(visitor)
    out.sort(key=lambda d: d["path"])
    return out

# -------- existing: parse_msr_to_tree (kept if you still use it elsewhere) --------

# -------- NEW: legacy helpers --------
def _logical_shape_str_for_field(n_rows: int, subshape: tuple) -> str:
    N = n_rows if (isinstance(n_rows, int) and n_rows > 0) else "N"
    if not subshape:
        return f"{N} × 1"
    dims = " × ".join(str(d) for d in subshape)
    return f"{dims} × {N}"

def _dtype_base_and_shape(dt: np.dtype):
    base = getattr(dt, "base", dt)
    shape = tuple(getattr(dt, "shape", ()) or ())
    return base, shape

def _dtype_fields_tree(dt: np.dtype, n_rows: int):
    """
    Recursively convert a structured dtype into a nested field tree.
    Each node: {name, dtype, shape, logical_shape, kind, children?}
    """
    names = getattr(dt, "names", None)
    if not names:
        return []
    out = []
    for name in names:
        
        sub_dt = dt.fields[name][0]
        base, subshape = _dtype_base_and_shape(sub_dt)
        node = {
            "name": name,
            "dtype": str(base),
            "shape": subshape,
            "logical_shape": _logical_shape_str_for_field(n_rows, subshape),
        }
        
        print("")
        shape = getattr(sub_dt, "shape", None)
        print("debug: dtype_fileds_tree: name: " + name)
        print("debug: dtype_fileds_tree: dtype: " + str(base))
        print("debug: dtype_fileds_tree: shape: " + str(shape))
        print("")
        
        if getattr(sub_dt, "names", None):          # nested struct
            node["kind"] = "struct"
            node["children"] = _dtype_fields_tree(sub_dt, n_rows)
        else:
            node["kind"] = "field"
        out.append(node)
    # mark “singleton root” if there is exactly one field and it is a struct (e.g., itr)
    if len(out) == 1 and out[0].get("kind") == "struct":
        out[0]["kind"] = "struct-root"
    return out



def _series_name_from_xml(xml_text: str) -> Optional[str]:
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)
        for el in root.iter():
            tag = el.tag.rsplit("}", 1)[-1].lower()
            if tag == "name" and el.text:
                return el.text.strip()
    except Exception:
        pass
    return None

def _try_load_legacy_mfx(msr_file: str, log=print):
    """Try to fetch MF/data {0} from legacy OBF-style MSR via msr-reader."""
    try:
        from msr_reader import OBFFile
    except Exception as e:
        log(f"[legacy] msr-reader not available: {e}")
        return None

    import numpy as np
    with OBFFile(msr_file) as obf:
        n = len(obf.shapes)
        target = None
        for i in range(n):
            try:
                xml = obf.get_imspector_xml_metadata(i)
            except Exception:
                xml = ""
            name = _series_name_from_xml(xml) or ""
            if "mf/data" in name.lower():
                target = i; break
        if target is None:
            # fallback: pick longest 1-D
            best_len = 0
            for i in range(n):
                a = np.asarray(obf.read_stack(i))
                if a.ndim == 1 and a.size > best_len:
                    best_len = a.size; target = i
        if target is None:
            return None
        arr = obf.read_stack(target)   # NumPy 1-D
        return arr

def _extract_legacy_reader_series(msr_file: str, log=print) -> dict:
    """
    Use msr-reader to enumerate legacy OBF stacks with a friendly 'StackSizes(...)' repr.
    Returns: {"count": int, "entries": [{"index": i, "name": str, "shape_repr": str}, ...]}
    """
    out = {"count": 0, "entries": []}
    try:
        from msr_reader import OBFFile
    except Exception as e:
        log(f"[legacy] msr-reader not available: {e}")
        return out

    with OBFFile(msr_file) as obf:
        shapes = getattr(obf, "shapes", [])
        n = len(shapes)
        out["count"] = n
        for i in range(n):
            try:
                xml = obf.get_imspector_xml_metadata(i)
            except Exception:
                xml = ""
            name = _series_name_from_xml(xml) or getattr(shapes[i], "name", f"Series_{i+1}")
            shape_repr = repr(shapes[i])  # typically "StackSizes(name='...', sizes=[...], ...)"
            out["entries"].append({
                "index": i,
                "name": name,
                "shape_repr": shape_repr,
            })
    return out



def _extract_legacy_series_from_metadata(meta) -> list:
    """
    Extract legacy 'series' from the OME-style metadata dict produced by specpy.File.metadata().
    Returns a list of dicts: {index, name, shape_str, dtype, kind}
    """
    out = []
    try:
        if not isinstance(meta, dict):
            return out
        ome_list = meta.get("OME")
        if not (isinstance(ome_list, list) and ome_list):
            return out
        ome0 = ome_list[0]
        root = ome0.get("") if isinstance(ome0, dict) else {}
        images = root.get("Image") or []
        for i, img in enumerate(images):
            inner = img.get("") if isinstance(img, dict) else {}
            name = inner.get("Name") or img.get("Name") or f"Series {i+1}"
            pixels_list = inner.get("Pixels") or []
            px = pixels_list[0] if pixels_list else {}
            pxd = px.get("") if isinstance(px, dict) else {}
            sx = pxd.get("SizeX", 0)
            sy = pxd.get("SizeY", 0)
            ptype = pxd.get("Type", "")
            # shape string like "1525 x 1275" or "668826202 x 1"
            shape_str = f"{sx} x {sy}" if sx and sy else f"{sx} x {sy}"
            # heuristic kind: wide/tall => image, 1D-ish => data
            kind = "image" if (isinstance(sx, int) and isinstance(sy, int) and sx > 1 and sy > 1) else "data"
            out.append({
                "index": i,
                "name": name,
                "shape_str": shape_str,
                "dtype": str(ptype),
                "kind": kind,
            })
    except Exception:
        # be permissive; just return what we have
        pass
    return out


def _dtypes_from_metadata(meta) -> list:
    """
    Extract per-series dtype strings from OME metadata produced by specpy.File.metadata().
    Returns a list dtype_by_index[i] = "uint8"/"int16"/"float32"/...
    """
    dtypes = []
    try:
        if not isinstance(meta, dict):
            return dtypes
        ome_list = meta.get("OME")
        if not (isinstance(ome_list, list) and ome_list):
            return dtypes
        ome0 = ome_list[0]
        root = ome0.get("") if isinstance(ome0, dict) else {}
        images = root.get("Image") or []
        for img in images:
            inner = img.get("") if isinstance(img, dict) else {}
            pixels_list = inner.get("Pixels") or []
            pxd = (pixels_list[0].get("") if pixels_list else {}) if isinstance(pixels_list[0], dict) else {}
            dt = pxd.get("Type", "")
            dtypes.append(str(dt) if dt is not None else "")
    except Exception:
        pass
    return dtypes


def _build_legacy_series_tree_entries(msr_file: str, meta) -> list[dict]:
    """
    Combine msr-reader 'shapes' (for sizes + internal name) and OME metadata (for dtype)
    to produce entries suitable for the UI tree.

    Each entry: { index, display_name, shape_str, dtype }
    """
    entries: list[dict] = []
    # dtype by index via metadata
    dtype_by_idx = _dtypes_from_metadata(meta)

    try:
        from msr_reader import OBFFile
    except Exception:
        # no msr-reader: we can still use metadata-only fallback
        # produce entries using OME Image/Name and SizeX/SizeY if present
        try:
            if not isinstance(meta, dict):
                return entries
            ome_list = meta.get("OME")
            if not (isinstance(ome_list, list) and ome_list):
                return entries
            images = (ome_list[0].get("") or {}).get("Image") or []
            for i, img in enumerate(images):
                inner = img.get("") if isinstance(img, dict) else {}
                name = inner.get("Name") or f"Series {i+1}"
                name_disp = str(name).replace("_", " ")
                pixels_list = inner.get("Pixels") or []
                pxd = (pixels_list[0].get("") if pixels_list else {}) if isinstance(pixels_list[0], dict) else {}
                sx, sy = pxd.get("SizeX", 0), pxd.get("SizeY", 0)
                if sx and sy:
                    shape_str = f"{sy} x {sx}"  # show as rows x cols
                else:
                    shape_str = f"{sx} x {sy}"
                dtype = dtype_by_idx[i] if i < len(dtype_by_idx) else ""
                entries.append({
                    "index": i,
                    "display_name": name_disp,
                    "shape_str": shape_str,
                    "dtype": dtype,
                })
            return entries
        except Exception:
            return entries

    # Full path with msr-reader
    with OBFFile(msr_file) as obf:
        shapes = getattr(obf, "shapes", [])
        for i, sh in enumerate(shapes):
            # name: prefer XML Name, else shape name
            try:
                xml = obf.get_imspector_xml_metadata(i)
            except Exception:
                xml = ""
            xml_name = _series_name_from_xml(xml)
            shape_name = getattr(sh, "name", f"Series_{i+1}")
            disp = (xml_name or shape_name or f"Series {i+1}").replace("_", " ")

            # sizes -> "rows x cols" (if 2D) else "N x 1"
            sizes = getattr(sh, "sizes", None)
            if isinstance(sizes, (list, tuple)) and len(sizes) >= 2:
                shape_str = f"{sizes[0]} x {sizes[1]}"
            elif isinstance(sizes, (list, tuple)) and len(sizes) == 1:
                shape_str = f"{sizes[0]} x 1"
            else:
                shape_str = str(sizes) if sizes is not None else ""

            dtype = dtype_by_idx[i] if i < len(dtype_by_idx) else ""
            entries.append({
                "index": i,
                "display_name": disp,
                "shape_str": shape_str,
                "dtype": dtype,
            })
    return entries



# -------- NEW: unified parse for UI --------
def parse_msr_general(msr_file: str, tmp_dir: str, log=print) -> Dict[str, Any]:
    """
    Unified parse for UI:
      - If modern: unpack all datasets to Zarr, build trees; set globals mfx/mbm from the FIRST dataset.
      - If legacy: print metadata, try to set mfx from MF/data {0} (if msr-reader present).
    Returns:
      {
        "mode": "modern"|"legacy",
        "msr": "<path>",
        # modern only:
        "datasets": [ {did, display_name, zroot, fields:list} , ... ],
        # legacy only:
        "metadata": dict|None,
      }
    """
    reset_state()
    msr_path = Path(msr_file)
    out_root = Path(tmp_dir) / f"msr_{msr_path.stem}"
    out_root.mkdir(parents=True, exist_ok=True)

    # load msr file with SpecPy package
    f = specpy.File(str(msr_file), specpy.File.Read)

    # Try modern
    try:
        info = f.minflux_datasets() or []               # modern format
    except Exception as e:
        log(f"[warn] minflux_datasets() failed: {e}")   # legacy format
        info = []

    if info:
        log(f"[parse] modern MINFLUX file: {len(info)} dataset(s)")
        datasets = []
        first_mfx = None
        first_mbm = None
        for i, ds in enumerate(info):       # index, dataset
            did = ds.get("did")
            name = ds.get("name") or ds.get("label") or str(did)
            key = name or str(did)  # <-- use as map key
            ds_dir = out_root / slug(name)
            ds_dir.mkdir(parents=True, exist_ok=True)       # zarr folder in temp folder
            log(f"  ds#{i} did={did} name={name} -> {ds_dir}")
            f.unpack(did, str(ds_dir))                      # export zarr data to zarr folder
            zroot = ds_dir / "zarr"                         # retrieve the zarr root folder path

            fields = []
            if zroot.is_dir():
                try:
                    fields = collect_zarr_fields(str(zroot))
                except Exception as e:
                    log(f"    [warn] collect fields failed: {e}")

                # NEW: load arrays into state maps (per dataset)
                try:
                    arch = zarr.open(str(zroot), mode="r")
                    if "mfx" in arch:
                        arr = arch["mfx"][:]  # NOTE: loads into RAM
                        from .state import set_mfx_for
                        set_mfx_for(key, arr)
                        log(f"    [mfx] loaded: key='{key}' shape={arr.shape} dtype={arr.dtype}")
                    if "grd/mbm/points" in arch:
                        arr = arch["grd/mbm/points"][:]
                        from .state import set_mbm_for
                        set_mbm_for(key, arr)
                        log(f"    [mbm] loaded: key='{key}' shape={arr.shape} dtype={arr.dtype}")
                except Exception as e:
                    log(f"    [warn] array load failed: {e}")

            datasets.append({
                "did": str(did),
                "display_name": name,
                "zroot": str(zroot),
                "fields": fields,
            })
        if first_mfx is not None:
            set_mfx(first_mfx)
            log(f"  [mfx] loaded into memory: shape={first_mfx.shape}, dtype={first_mfx.dtype}")
        if first_mbm is not None:
            set_mbm(first_mbm)
            log(f"  [mbm] loaded into memory: shape={first_mbm.shape}, dtype={first_mbm.dtype}")

        return {"mode": "modern", "msr": str(msr_file), "datasets": datasets}

    # Legacy path
    log("[parse] legacy/OBF-style file (no minflux_datasets)")
    try:
        meta = f.metadata()
        log("OME-XML metadata loaded; showing in UI")
    except Exception as e:
        meta = None
        log(f"[warn] metadata() failed: {e}")

    # Try to load MF/data as mfx equivalent (optional)
    legacy_mfx = _try_load_legacy_mfx(str(msr_file), log=log)
    if legacy_mfx is not None:
        try:
            from .state import set_mfx_for
            set_mfx_for("legacy", legacy_mfx)
        except Exception:
            pass
        log(f"  [legacy mfx] 1-D vector loaded: shape={legacy_mfx.shape}, dtype={legacy_mfx.dtype}")

    # Build entries for the UI tree (Series rows with shape + dtype)
    legacy_series_tree = _build_legacy_series_tree_entries(str(msr_file), meta)

    return {
        "mode": "legacy",
        "msr": str(msr_file),
        "metadata": meta,
        "legacy_series_tree": legacy_series_tree,  # NEW
    }

# --- Back-compat wrapper: prefer parse_msr_general() ---
def parse_msr_to_tree(msr_file: str, tmp_dir: str, log=print):
    """
    Back-compat shim. Old code imported parse_msr_to_tree; we now route to
    parse_msr_general and return its dict so imports stop breaking.

    Return value:
      dict with keys: "mode", "msr", and depending on mode:
        - modern:  "datasets": [ {did, display_name, zroot, fields}, ... ]
        - legacy:  "metadata": dict
    """
    return parse_msr_general(msr_file, tmp_dir, log=log)


# --- Back-compat: find a single .msr from a path (file or folder) ---
def pick_one_msr(input_path: Optional[str]) -> Optional[str]:
    """
    Return a single .msr path from `input_path`.
      - If input_path is an .msr file, return it.
      - If it's a directory, return the first *.msr inside (sorted).
      - If None/empty/invalid, return None.
    """
    if not input_path:
        return None
    try:
        p = Path(input_path)
    except Exception:
        return None

    if p.is_file() and p.suffix.lower() == ".msr":
        return str(p)

    if p.is_dir():
        # Pick the first *.msr (case-insensitive) in a stable order
        candidates = sorted(list(p.glob("*.msr")) + list(p.glob("*.MSR")))
        if candidates:
            return str(candidates[0])

    return None
