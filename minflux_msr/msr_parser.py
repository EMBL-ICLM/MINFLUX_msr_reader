# msr_parser.py
import os
from pathlib import Path
import zarr
import numpy as np
from specpy import File as SpecFile
from msr_reader import OBFFile
from .parser_model import DatasetInfo, FieldInfo

# Helper: logical shape formatter
def _logical_shape(n_rows: int, subshape: tuple) -> str:
    N = n_rows if isinstance(n_rows, int) and n_rows > 0 else "N"
    if not subshape:
        return f"{N} × 1"
    return " × ".join(str(d) for d in subshape) + f" × {N}"

def _dtype_fields(dt: np.dtype, n_rows: int) -> list[FieldInfo]:
    names = getattr(dt, "names", None)
    if not names:
        return []
    out = []
    for name in names:
        subdt = dt.fields[name][0]
        base = getattr(subdt, "base", subdt)
        shape = getattr(subdt, "shape", ())
        logical = _logical_shape(n_rows, shape)
        children = _dtype_fields(subdt, n_rows) if getattr(subdt, "names", None) else []
        kind = "struct" if children else "field"
        out.append(FieldInfo(name=name, dtype=str(base), shape=shape, logical_shape=logical, kind=kind, children=children))
    # singleton root
    if len(out) == 1 and out[0].kind == "struct":
        out[0].kind = "struct-root"
    return out

def parse_modern(msr_path: str, tmp: str, log=print) -> list[DatasetInfo]:
    sf = SpecFile(msr_path, SpecFile.Read)
    info = sf.minflux_datasets() or []
    result = []
    for ds in info:
        did = ds.get("did")
        name = ds.get("name") or str(did)
        out_dir = Path(tmp) / name
        sf.unpack(did, str(out_dir))
        grp = zarr.open(str(out_dir / "zarr"), mode="r")
        fields = []
        
        # traverse a data node, for group, expand; for array, parse and append to fields:
        def traverse(node, path=""):

            if isinstance(node, zarr.Array): # node is Zarr Array: /mfx
                n_rows = node.shape[0] if node.shape else 0
                dtype = node.dtype
                
                for attr_name in dtype.names: # 1st layer fields below mfx

                    data_attr = node[attr_name].dtype
                    print("data attr name: " + attr_name)
                    
                    if data_attr.names: # nested fields (itr), None otherwise
                        for fd_name in data_attr.names:
                            print("data attr subfield name: " + fd_name )

                if dtype.names: # structured
                    children = _dtype_fields(dtype, n_rows)
                    fields.append(FieldInfo(name=path, dtype=str(dtype), shape=node.shape, logical_shape="", kind="structured", children=children))
                else:           # array
                    fields.append(FieldInfo(name=path, dtype=str(dtype), shape=node.shape, logical_shape="", kind="array", children=[]))

            elif isinstance(node, zarr.Group): # node is Zarr Group: /(dataset), /grd, /mbm
                for k in node:      # /grd;     /grd/mbm;     /mfx
                    traverse(node[k], f"{path}/{k}" if path else k)

        traverse(grp)
        result.append(DatasetInfo(did=did, name=name, root=str(out_dir), fields=fields))
    return result

def parse_legacy(msr_path: str, tmp: str, log=print) -> list[DatasetInfo]:
    result = []
    with OBFFile(msr_path) as obf:
        n = len(obf.series)
        fields = []
        for i in range(n):
            shape = obf.shapes[i].sizes
            dtype = obf.dtype(i)
            name = obf.series[i].get("name", "") or f"Series_{i+1}"
            fields.append(FieldInfo(name=str(name), dtype=str(dtype), shape=tuple(shape), logical_shape="", kind="legacy", children=[]))
    return [DatasetInfo(did=msr_path, name=Path(msr_path).stem, root="", fields=fields)]

def flatten(dt):
    
    return []