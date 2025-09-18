# ui_tk/app.py
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from minflux_msr import (
    slug, pick_one_msr, parse_msr_general,
    export_arrays,
)
from minflux_msr import state as MFSTATE
from .field_dialog import FieldDialog

EXPORT_MODES = [
    "mfx",
    "mbm",
    "mfx+mbm separately",
    "mfx+mbm combined",
]

def parse_export_mode(mode_str: str):
    mode = mode_str.lower().strip()
    if mode == "mfx": return True, False, False
    if mode == "mbm": return False, True, False
    if mode == "mfx+mbm separately": return True, True, False
    if mode == "mfx+mbm combined": return True, True, True
    return True, False, False

def _is_numeric_dtype(dt) -> bool:
    try:
        import numpy as np
        return np.issubdtype(dt, np.integer) or np.issubdtype(dt, np.floating)
    except Exception:
        return False

def _save_tiff(arr, out_path: Path, log=print):
    import numpy as np
    import tifffile as tiff
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {a.shape}")
    if a.dtype.kind == "f":
        a = a.astype(np.float32, copy=False)
    elif a.dtype.itemsize == 1:
        a = a.astype(np.uint8, copy=False)
    elif a.dtype.itemsize == 2:
        a = a.astype(np.uint16, copy=False)
    elif a.dtype.itemsize == 4 and a.dtype.kind == "i":
        a = a.astype(np.int32, copy=False)
    tiff.imwrite(str(out_path), a)
    log(f"[tiff] wrote {out_path}")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MINFLUX .msr Parser & Exporter")
        # a little thinner, a little taller
        self.geometry("650x780")

        # Inputs
        self.mode = tk.StringVar(value="file")
        self.input_path = tk.StringVar(value="")
        self.tmp_dir = tk.StringVar(value=r"C:/data/temp")
        self.out_dir = tk.StringVar(value=r"C:/data/temp")  # Output folder (moved below tree)

        # Preview rows control
        self.preview_rows = tk.IntVar(value=100)

        # Datasets / Fields included... options
        self.field_selection = {}  # key -> {"checked": bool, "mfx": set|None, "mbm": set|None}
        
        # Export format checkboxes
        self.fmt_mat = tk.BooleanVar(value=True)
        self.fmt_npy = tk.BooleanVar(value=False)
        self.fmt_npz = tk.BooleanVar(value=False)
        self.fmt_json = tk.BooleanVar(value=False)
        self.fmt_csv = tk.BooleanVar(value=False)

        # Parsed cache & node info
        self.parsed: Dict[str, Any] = {}
        self._nodeinfo: Dict[str, Dict[str, Any]] = {}         # generic per-node info
        self._fullpath_by_node: Dict[str, str] = {}            # zarr path for modern nodes
        self._datasetnode_info: Dict[str, Dict[str, Any]] = {} # zroot per dataset node

        self.PREVIEW_N_DEFAULT = 100

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        # Row: input type
        row = ttk.Frame(self); row.pack(fill="x", **pad)
        ttk.Label(row, text="Input type:").pack(side="left")
        ttk.Radiobutton(row, text="Single .msr file", variable=self.mode, value="file").pack(side="left", padx=8)
        ttk.Radiobutton(row, text="Folder (batch)", variable=self.mode, value="folder").pack(side="left", padx=8)

        # Row: input path
        row = ttk.Frame(self); row.pack(fill="x", **pad)
        ttk.Label(row, text="MSR file or folder:").pack(side="left")
        ttk.Entry(row, textvariable=self.input_path).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row, text="Browse…", command=self.browse_input).pack(side="left")

        # Row: temp dir
        row = ttk.Frame(self); row.pack(fill="x", **pad)
        ttk.Label(row, text="Temp directory (Zarr + caches):").pack(side="left")
        ttk.Entry(row, textvariable=self.tmp_dir).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row, text="Browse…", command=self.browse_tmp).pack(side="left")

        # Row: actions (Parse + Preview rows)
        row = ttk.Frame(self); row.pack(fill="x", **pad)
        ttk.Button(row, text="Parse MSR file", command=self.on_parse).pack(side="left")
        ttk.Label(row, text="Preview rows:").pack(side="left", padx=(16, 6))
        pr = ttk.Entry(row, width=8, textvariable=self.preview_rows)
        pr.pack(side="left")

        # Tree: Parsed content
        tvf = ttk.LabelFrame(self, text="Parsed content")
        tvf.pack(fill="both", expand=True, **pad)
        self.tree = ttk.Treeview(tvf, columns=("kind","shape","dtype"), show="tree headings", height=9)
        self.tree.heading("#0", text="Path / Info")
        self.tree.heading("kind", text="Kind")
        self.tree.heading("shape", text="Shape")
        self.tree.heading("dtype", text="DType")
        self.tree.column("#0", width=140); self.tree.column("kind", width=30)
        self.tree.column("shape", width=80); self.tree.column("dtype", width=160)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewOpen>>", self.on_tree_open)
        
        # Right-click context menu
        self.tree.bind("<Button-3>", self._on_tree_right_click)
        self._attach_context_menu()

        # Moved: Output folder row (below tree)
        row = ttk.Frame(self); row.pack(fill="x", **pad)
        ttk.Label(row, text="Output folder:").pack(side="left")
        ttk.Entry(row, textvariable=self.out_dir).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row, text="Browse…", command=self.browse_out).pack(side="left")

        # Export row: Fields button + formats
        exp = ttk.Frame(self); exp.pack(fill="x", **pad)
        ttk.Button(exp, text="Datasets / Fields included…", command=self.on_fields_dialog).pack(side="left")
        ttk.Label(exp, text="Formats:").pack(side="left", padx=(10, 6))
        ttk.Checkbutton(exp, text="MATLAB (.mat)", variable=self.fmt_mat).pack(side="left", padx=6)
        ttk.Checkbutton(exp, text="NumPy (.npy)", variable=self.fmt_npy).pack(side="left", padx=6)
        ttk.Checkbutton(exp, text="JSON (.json)", variable=self.fmt_json).pack(side="left", padx=6)
        ttk.Checkbutton(exp, text="(.csv)", variable=self.fmt_csv).pack(side="left", padx=6)

        # Log
        logf = ttk.LabelFrame(self, text="Log")
        logf.pack(fill="both", expand=False, **pad)
        self.logbox = ScrolledText(logf, height=10, wrap="word", font=("Consolas", 10)); self.logbox.pack(fill="both", expand=True)

        # Bottom buttons
        row = ttk.Frame(self); row.pack(fill="x", **pad)
        ttk.Button(row, text="OK (export)", command=self.on_ok).pack(side="right")
        ttk.Button(row, text="Cancel", command=self.destroy).pack(side="right", padx=8)

    # Browse helpers
    def browse_input(self):
        if self.mode.get() == "file":
            p = filedialog.askopenfilename(title="Choose an .msr file", filetypes=[("Imspector .msr", "*.msr"), ("All files", "*.*")])
        else:
            p = filedialog.askdirectory(title="Choose a folder of .msr files")
        if p: self.input_path.set(p)

    def browse_tmp(self):
        p = filedialog.askdirectory(title="Choose temp directory")
        if p:
            self.tmp_dir.set(p)
            if not self.out_dir.get():
                self.out_dir.set(p)

    def browse_out(self):
        p = filedialog.askdirectory(title="Choose output folder")
        if p: self.out_dir.set(p)

    # Logging
    def log(self, msg: str):
        self.logbox.insert("end", msg + "\n"); self.logbox.see("end"); self.logbox.update_idletasks()
        print(msg)

    # Parse
    def on_parse(self):
        # Clear UI + globals
        self.tree.delete(*self.tree.get_children())
        MFSTATE.reset()
        # NEW: clear node caches so fresh nodes get mapped
        self._nodeinfo.clear()
        self._fullpath_by_node.clear()
        self._datasetnode_info.clear()
    
        ipath = self.input_path.get().strip(); tmp = self.tmp_dir.get().strip()
        if not ipath or not os.path.exists(ipath) or not tmp:
            self.log("[error] set an input .msr (or folder) and a temp directory first.")
            return

        msr = ipath if os.path.isfile(ipath) else next((str(p) for p in Path(ipath).glob("*.msr")), None)
        if not msr:
            self.log("no msr file found!"); return

        self.log(f"[parse] using: {msr}")
        res = parse_msr_general(msr, tmp, log=self.log)
        self.parsed = res

        root_id = self.tree.insert("", "end", text=os.path.basename(msr), values=("", "", ""), open=True)

        if res.get("mode") == "modern":
            for ds in res.get("datasets", []):
                ds_id = self.tree.insert(root_id, "end",
                                         text=(ds.get("display_name") or ds.get("did")),
                                         values=("dataset", "", ds.get("did","")), open=True)
                self._datasetnode_info[ds_id] = {"did": ds.get("did"), "zroot": ds.get("zroot")}
                self._fullpath_by_node[ds_id] = ""

                node_map = {"": ds_id}
                for fld in (ds.get("fields") or []):
                    path = fld["path"]; parts = path.split("/")
                    cur = ""; parent_id = ds_id
                    for i, part in enumerate(parts):
                        cur = part if i == 0 else f"{cur}/{part}"
                        if cur in node_map:
                            parent_id = node_map[cur]; continue
                        is_last = (i == len(parts) - 1)
                        kind = fld["kind"] if is_last else "group"
                        shape = "" if kind != "array" else str(fld.get("shape"))
                        dtype = "" if kind != "array" else fld.get("dtype", "")
                        node_id = self.tree.insert(parent_id, "end",
                                                   text=part,
                                                   values=(kind, shape, dtype),
                                                   open=(cur in ("mfx", "grd", "grd/mbm")))
                        node_map[cur] = node_id
                        parent_id = node_id
                        self._fullpath_by_node[node_id] = cur
                        # Store info for right-click preview/export if this is an array
                        if is_last and kind == "array":
                            self._nodeinfo[node_id] = {"type":"modern_array", "zroot": ds["zroot"], "path": path}

                            # # dtype subfields
                            # for sub in fld.get("dtype_fields", []) or []:
                            #     sub_node = self.tree.insert(node_id, "end",
                            #                                 text=sub["name"],
                            #                                 values=("field", str(sub.get("shape") or ()), sub.get("dtype","")))
                            #     self._fullpath_by_node[sub_node] = f"{path}/{sub['name']}"
                            tree_fields = fld.get("dtype_fields") or []
                            if tree_fields:
                                self._add_dtype_field_nodes(node_id, path, tree_fields)

        else:
            # legacy: Series rows + <metadata>
            meta = res.get("metadata")
            series_rows = res.get("legacy_series_tree") or []

            series_root = self.tree.insert(root_id, "end",
                                           text="Series",
                                           values=("", "", ""),
                                           open=True)
            for s in series_rows:
                idx1 = int(s.get("index", 0)) + 1
                name = s.get("display_name", f"Series {idx1}")
                text = f"- Series {idx1}: {name}"
                shape_str = s.get("shape_str", "")
                dtype = s.get("dtype", "")
                node_id = self.tree.insert(series_root, "end",
                                           text=text,
                                           values=("", shape_str, dtype))
                # store legacy series info for context menu
                self._nodeinfo[node_id] = {"type":"legacy_series", "msr": msr, "series_index": idx1-1, "name": name}

            meta_root = self.tree.insert(root_id, "end",
                                         text="<metadata>",
                                         values=("", "", ""),
                                         open=False)
            if isinstance(meta, dict):
                import json
                pretty = json.dumps(meta, indent=2, sort_keys=True, default=str).splitlines()
                for line in pretty[:800]:
                    self.tree.insert(meta_root, "end", text=line, values=("", "", ""))
            else:
                self.tree.insert(meta_root, "end", text="(metadata unavailable)", values=("", "", ""))

        # Log what is in memory
        if getattr(MFSTATE, "mfx_map", {}):
            self.log(f"[global] mfx datasets: {len(MFSTATE.mfx_map)}")
            for k, a in MFSTATE.mfx_map.items():
                self.log(f"          - {k}: shape={a.shape}, dtype={a.dtype}")
        else:
            self.log("[global] mfx is empty")
        if getattr(MFSTATE, "mbm_map", {}):
            self.log(f"[global] mbm datasets: {len(MFSTATE.mbm_map)}")
            for k, a in MFSTATE.mbm_map.items():
                self.log(f"          - {k}: shape={a.shape}, dtype={a.dtype}")
        else:
            self.log("[global] mbm is empty")

    # Expand-trigger preview (kept for old preview nodes if any are added later)
    def on_tree_open(self, event=None):
        # We now preview via context menu; keep this stub in case
        pass

    # ---------- Right-click (context menu) ----------
    def _attach_context_menu(self):
        self._ctx_node = None
        self._ctx_menu = tk.Menu(self, tearoff=0)
        self._ctx_menu.add_command(label="Preview", command=self._ctx_preview)
        self._ctx_menu.add_command(label="Export to CSV…", command=self._ctx_export_csv)
        self.tree.bind("<Button-3>", self._on_tree_right_click)  # Windows / most X11
        self.tree.bind("<Control-Button-1>", self._on_tree_right_click)  # fallback

    def _on_tree_right_click(self, event):
        row = self.tree.identify_row(event.y)
        if not row:
            return
        self.tree.selection_set(row)
        self.tree.focus(row)
        self._ctx_node = row
        # enable/disable export based on node kind
        kind = self.tree.set(row, "kind")
        # Export allowed for leaf 'field' or plain numeric 'array'
        can_export = (kind == "field") or (kind == "array")
        self._ctx_menu.entryconfig("Export to CSV…", state="normal" if can_export else "disabled")
        try:
            self._ctx_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._ctx_menu.grab_release()


    def _add_dtype_field_nodes(self, parent_id: str, path_prefix: str, fields: list):
        """Recursively add dtype field nodes (supports nested structs)."""
        for sub in fields or []:
            text = sub.get("name", "")
            print("debug: App, add dtype field node name: " + text)
            kind = "struct" if sub.get("children") else sub.get("kind", "field")
            shape = sub.get("logical_shape") or str(sub.get("shape") or ())
            dtype = sub.get("dtype", "")
            node = self.tree.insert(parent_id, "end",
                                    text=text,
                                    values=(kind, shape, dtype))
            # Track fullpath for potential actions
            self._fullpath_by_node[node] = f"{path_prefix}/{text}" if path_prefix else text
            # Recurse if nested
            if sub.get("children"):
                self._add_dtype_field_nodes(node, f"{path_prefix}/{text}", sub["children"])


    def _dataset_node_of(self, node_id: str) -> Optional[str]:
        """Ascend to dataset node (whose 'kind' column == 'dataset')."""
        n = node_id
        while n:
            if self.tree.set(n, "kind") == "dataset":
                return n
            n = self.tree.parent(n)
        return None

    def _zroot_did_for_dataset_node(self, ds_node: str) -> Optional[tuple]:
        info = self._datasetnode_info.get(ds_node)
        if info:
            return info.get("zroot"), info.get("did")
        return None

    def _full_path_for_node(self, node_id: str) -> Optional[str]:
        return self._fullpath_by_node.get(node_id)

    def _load_array_for_path(self, zroot: str, path: str):
        """Return a NumPy view for 'path' or 'array/field' under zroot."""
        import numpy as np, zarr
        arch = zarr.open(zroot, mode="r")
        if path in arch:
            return np.asarray(arch[path])
        if "/" in path:
            parent, field = path.rsplit("/", 1)
            if parent in arch:
                arr = arch[parent]
                names = getattr(arr.dtype, "names", None)
                if names and (field in names):
                    return np.asarray(arr[field])
        raise KeyError(f"path not found: {path}")

    def _to_2d(self, a):
        import numpy as np
        a = np.asarray(a)
        if a.ndim == 1:
            return a.reshape(-1, 1)
        if a.ndim > 2:
            return a.reshape(a.shape[0], -1)
        return a

    def _ctx_preview(self):
        """Preview first N entries of selected field/array in a popup window."""
        node = self._ctx_node
        if not node:
            return
        ds_node = self._dataset_node_of(node)
        if not ds_node:
            return
        zroot, did = self._zroot_did_for_dataset_node(ds_node)
        if not zroot:
            return
        path = self._full_path_for_node(node)
        if not path:
            # If they right-clicked e.g. 'vld' subnode we stored that as path; if missing, bail
            messagebox.showinfo("Preview", "Choose a field or array node.")
            return

        import numpy as np
        try:
            arr = self._load_array_for_path(zroot, path)
        except Exception as e:
            messagebox.showerror("Preview error", f"Cannot load {path}:\n{e}")
            return

        N = min(int(self.preview_rows.get()), arr.shape[0] if hasattr(arr, "shape") and arr.ndim >= 1 else 0)
        a2 = self._to_2d(arr[:N])

        # Build a simple preview window
        win = tk.Toplevel(self)
        win.title(f"Preview: {path}  (N={arr.shape[0] if hasattr(arr,'shape') else '?'}, showing {N})")
        tv = ttk.Treeview(win, show="headings", height=min(20, N))
        # columns
        cols = [f"c{i}" for i in range(a2.shape[1] if N else 1)]
        tv["columns"] = cols
        for i, c in enumerate(cols):
            tv.heading(c, text=str(i))
            tv.column(c, width=90, anchor="e")
        # rows
        for r in range(N):
            tv.insert("", "end", values=[a2[r, i] for i in range(a2.shape[1])])
        tv.pack(fill="both", expand=True, padx=8, pady=8)
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0,8))

    def _ctx_export_csv(self):
        """Export current selected field (or plain numeric array) to CSV."""
        node = self._ctx_node
        if not node:
            return
        kind = self.tree.set(node, "kind")
        ds_node = self._dataset_node_of(node)
        if not ds_node:
            return
        zroot, did = self._zroot_did_for_dataset_node(ds_node)
        path = self._full_path_for_node(node)

        if not path:
            messagebox.showinfo("Export", "Choose a field or array node.")
            return

        import numpy as np, zarr
        arch = zarr.open(zroot, mode="r")
        # For arrays with structured dtype, require a sub-field; otherwise export the array itself
        if kind == "array":
            arr = arch[path]
            if getattr(arr.dtype, "names", None):
                messagebox.showwarning("Export",
                                    "This is a structured array. Please right-click a sub-field (e.g. mfx/vld).")
                return
            data = np.asarray(arr)
        else:  # 'field'
            try:
                data = self._load_array_for_path(zroot, path)
            except Exception as e:
                messagebox.showerror("Export error", f"Cannot load {path}:\n{e}")
                return

        # choose file
        default = f"{Path(zroot).parent.name}_{path.replace('/','_')}.csv"
        fname = filedialog.asksaveasfilename(
            title="Export to CSV",
            defaultextension=".csv",
            initialfile=default,
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if not fname:
            return

        try:
            np.savetxt(fname, self._to_2d(data), delimiter=",")
            self.log(f"[export] wrote {fname}")
            messagebox.showinfo("Export", f"Saved:\n{fname}")
        except Exception as e:
            messagebox.showerror("Export error", f"Failed to save CSV:\n{e}")

    # Fields dialog placeholder (kept)
    def _gather_datasets_for_dialog(self) -> List[dict]:
        """Build dataset/field info from current state maps for the dialog."""
        datasets = []
        # Prefer modern: use parsed datasets for stable ordering/names
        if self.parsed.get("mode") == "modern":
            for ds in (self.parsed.get("datasets") or []):
                key = ds.get("display_name") or ds.get("did") or "dataset"
                # derive fields from loaded arrays (robust to versioning)
                mfx_arr = MFSTATE.mfx_map.get(key)
                mbm_arr = MFSTATE.mbm_map.get(key)
                mfx_fields = list(getattr(getattr(mfx_arr, "dtype", None), "names", []) or [])
                mbm_fields = list(getattr(getattr(mbm_arr, "dtype", None), "names", []) or [])
                datasets.append({"key": key, "name": key, "mfx_fields": mfx_fields, "mbm_fields": mbm_fields})
        else:
            # legacy: expose a synthetic 'legacy' dataset with whatever we have
            key = "legacy"
            mfx_arr = MFSTATE.mfx_map.get(key) or MFSTATE.mfx
            mbm_arr = MFSTATE.mbm_map.get(key) or MFSTATE.mbm
            mfx_fields = list(getattr(getattr(mfx_arr, "dtype", None), "names", []) or [])
            mbm_fields = list(getattr(getattr(mbm_arr, "dtype", None), "names", []) or [])
            datasets.append({"key": key, "name": key, "mfx_fields": mfx_fields, "mbm_fields": mbm_fields})
        return datasets

    def on_fields_dialog(self):
        ds_list = self._gather_datasets_for_dialog()
        if not ds_list:
            from tkinter import messagebox
            messagebox.showinfo("No datasets", "No datasets loaded to select fields from.")
            return
        dlg = FieldDialog(self, ds_list, prechecked=self.field_selection or None)
        if dlg.result is not None:
            self.field_selection = dlg.result
            self.log("[select] updated datasets/fields selection.")

    # helpers for batch export option
    def _derive_global_field_filters(self):
        """
        Merge the current field_selection across datasets into a single pair
        (mfx_sel, mbm_sel) to apply uniformly in batch mode.
        - If any dataset had 'all' (None), we return None for that type (=> all).
        - Else we return the union of selected fields.
        """
        sel = self.field_selection or {}
        if not sel:
            return None, None

        # MFX
        any_all_mfx = any(v.get("mfx") is None for v in sel.values())
        if any_all_mfx:
            mfx_sel = None
        else:
            mfx_union = set()
            for v in sel.values():
                m = v.get("mfx")
                if m: mfx_union |= set(m)
            mfx_sel = mfx_union or None  # empty => treat as all

        # MBM
        any_all_mbm = any(v.get("mbm") is None for v in sel.values())
        if any_all_mbm:
            mbm_sel = None
        else:
            mbm_union = set()
            for v in sel.values():
                m = v.get("mbm")
                if m: mbm_union |= set(m)
            mbm_sel = mbm_union or None

        return mfx_sel, mbm_sel


    def _export_current_parsed(self, out_dir: str, formats: list, mfx_sel_global=None, mbm_sel_global=None):
        """
        Export the datasets currently loaded in MFSTATE.*_map into out_dir,
        honoring either a global field selection (for batch) or the per-dataset
        self.field_selection (for single-file).
        """
        import numpy as np
        from pathlib import Path

        os.makedirs(out_dir, exist_ok=True)

        def subset_struct(arr, names_sel):
            if arr is None:
                return None
            names = getattr(getattr(arr, "dtype", None), "names", None)
            if names and names_sel:
                keep = [n for n in names if n in names_sel]
                if keep:
                    try:
                        return arr[keep]
                    except Exception:
                        pass
            return arr

        def save_mat(fn_base: str, arr: "np.ndarray|None"):
            if arr is None: return None
            try:
                from scipy.io import savemat
            except Exception as e:
                self.log(f"[warn] SciPy not available; skip .mat ({e})")
                return None
            def as_col(a):
                a = np.asarray(a)
                if a.ndim == 1: return a.reshape(-1, 1)
                if a.ndim == 2: return a
                return a.reshape(a.shape[0], -1)
            payload = {}
            names = getattr(getattr(arr, "dtype", None), "names", None)
            if names:
                for k in names:
                    col = arr[k]
                    subnames = getattr(col.dtype, "names", None)
                    if subnames:
                        for sk in subnames:
                            payload[f"{k}_{sk}"] = as_col(col[sk])
                    else:
                        payload[k] = as_col(col)
            else:
                payload["data"] = as_col(arr)
            path = Path(out_dir) / f"{fn_base}.mat"
            try:
                savemat(str(path), payload, do_compression=False, oned_as="column", long_field_names=True)
                self.log(f"[mat] wrote {path}")
                return str(path)
            except Exception as e:
                self.log(f"[error] mat save failed: {e}")
                return None

        def save_npy(fn_base: str, arr: "np.ndarray|None"):
            if arr is None: return None
            path = Path(out_dir) / f"{fn_base}.npy"
            try:
                np.save(str(path), arr, allow_pickle=False)
                self.log(f"[npy] wrote {path}")
                return str(path)
            except Exception as e:
                self.log(f"[error] npy save failed: {e}")
                return None

        def save_json(fn_base: str, arr: "np.ndarray|None"):
            if arr is None: return None
            names = getattr(getattr(arr, "dtype", None), "names", None)
            if not names:
                self.log("[json] skipping (no named fields).")
                return None
            wanted_order = [
                "vld","fnl","bot","eot",
                "sta","tim","tid","gri","thi","sqi","itr",
                "loc","lnc","eco","ecc","efo","efc","fbg","cfr","dcr"
            ]
            fields_present = [k for k in wanted_order if k in names] or list(names)
            try:
                import orjson as _json
                dumps = lambda o: _json.dumps(o, option=_json.OPT_SERIALIZE_NUMPY)
            except Exception:
                import json as _json
                dumps = lambda o: _json.dumps(o, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

            path = Path(out_dir) / f"{fn_base}.json"
            N = int(arr.shape[0]) if getattr(arr, "shape", None) else 0
            CHUNK = 100_000
            with open(path, "wb") as fh:
                fh.write(b"[\n"); first = True
                for start in range(0, N, CHUNK):
                    stop = min(start + CHUNK, N)
                    chunk = arr[start:stop][fields_present]
                    recs = chunk.tolist()
                    objs = []
                    for tpl in recs:
                        d = {}
                        for k, v in zip(fields_present, tpl):
                            if isinstance(v, np.ndarray): d[k] = v.tolist()
                            elif isinstance(v, np.generic): d[k] = v.item()
                            else: d[k] = v
                        objs.append(d)
                    payload = dumps(objs)
                    if payload.startswith(b"[") and payload.endswith(b"]"): payload = payload[1:-1]
                    if not first and payload: fh.write(b",\n")
                    fh.write(payload); first = False
                fh.write(b"\n]\n")
            self.log(f"[json] wrote {path}")
            return str(path)

        def save_csv(fn_base: str, arr: "np.ndarray|None"):
            if arr is None: return None
            names = getattr(getattr(arr, "dtype", None), "names", None)
            if not names:
                self.log("[csv] skipping (no named fields).")
                return None
            import csv
            path = Path(out_dir) / f"{fn_base}.csv"
            try:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(list(names))
                    for row in arr:
                        w.writerow([np.asarray(row[k]).tolist() if hasattr(row[k], "shape") else row[k] for k in names])
                self.log(f"[csv] wrote {path}")
                return str(path)
            except Exception as e:
                self.log(f"[error] csv save failed: {e}")
                return None

        # Iterate datasets in UI order if modern; otherwise by map order
        if self.parsed.get("mode") == "modern":
            datasets = [ (ds.get("display_name") or ds.get("did") or "dataset") for ds in (self.parsed.get("datasets") or []) ]
        else:
            datasets = list(getattr(MFSTATE, "mfx_map", {}).keys() | getattr(MFSTATE, "mbm_map", {}).keys()) or ["legacy"]

        for key in datasets:
            base = key  # dataset base name
            mfx_arr = MFSTATE.mfx_map.get(key)
            mbm_arr = MFSTATE.mbm_map.get(key)

            # Decide field selections: batch uses global; single-file uses per-dataset selection
            if mfx_sel_global is not None or mbm_sel_global is not None:
                mfx_sel = mfx_sel_global
                mbm_sel = mbm_sel_global
            else:
                sel = self.field_selection.get(key, {"checked": True, "mfx": None, "mbm": None})
                if not sel.get("checked", True):
                    self.log(f"[export] skip dataset '{key}' (unchecked).")
                    continue
                mfx_sel = sel.get("mfx")
                mbm_sel = sel.get("mbm")

            mfx_sub = subset_struct(mfx_arr, mfx_sel)
            mbm_sub = subset_struct(mbm_arr, mbm_sel)

            # mfx (separate files)
            fn_mfx = f"{base}_mfx"
            if "mat" in formats:  save_mat(fn_mfx, mfx_sub)
            if "npy" in formats:  save_npy(fn_mfx, mfx_sub)
            if "json" in formats: save_json(fn_mfx, mfx_sub)
            if "csv" in formats:  save_csv(fn_mfx, mfx_sub)

            # mbm (separate files)
            fn_mbm = f"{base}_mbm"
            if "mat" in formats:  save_mat(fn_mbm, mbm_sub)
            if "npy" in formats:  save_npy(fn_mbm, mbm_sub)
            if "json" in formats: save_json(fn_mbm, mbm_sub)
            if "csv" in formats:  save_csv(fn_mbm, mbm_sub)

    def _find_msr_files(self, path_str: str):
        """Return a sorted list of Path objects for all *.msr in a folder.
        If a file path is given by mistake, use its parent folder."""
        from pathlib import Path
        p = Path(path_str)
        folder = p.parent if p.is_file() else p
        files = sorted(list(folder.glob("*.msr")) + list(folder.glob("*.MSR")))
        return files
    
    # Main export button
    def on_ok(self):
        out_dir = self.out_dir.get().strip() or self.tmp_dir.get().strip()
        if not out_dir:
            from tkinter import messagebox
            messagebox.showerror("Missing output", "Please set an output folder.")
            return
        os.makedirs(out_dir, exist_ok=True)

        # Formats (no NPZ combined)
        formats = []
        if self.fmt_mat.get():  formats.append("mat")
        if self.fmt_npy.get():  formats.append("npy")
        if self.fmt_json.get(): formats.append("json")
        if self.fmt_csv.get():  formats.append("csv")
        if not formats:
            from tkinter import messagebox
            messagebox.showerror("No formats", "Pick at least one export format.")
            return

        tmp = self.tmp_dir.get().strip()
        ipath = self.input_path.get().strip()

        # ---------- BATCH MODE ----------
        if self.mode.get() == "folder":
            files = self._find_msr_files(ipath)
            if not files:
                self.log("[batch] no .msr files found in the folder.")
                return

            # Merge current field selections into global filters for batch
            mfx_sel_global, mbm_sel_global = self._derive_global_field_filters()
            self.log(f"[batch] exporting {len(files)} files with global filters: "
                    f"mfx={'ALL' if mfx_sel_global is None else sorted(mfx_sel_global)}; "
                    f"mbm={'ALL' if mbm_sel_global is None else sorted(mbm_sel_global)}")

            from pathlib import Path
            from minflux_msr import slug  # reuse your slug sanitizer

            for idx, msr_path in enumerate(files, 1):
                try:
                    # Make a subfolder per file
                    sub_out = Path(out_dir) / slug(msr_path.stem)
                    os.makedirs(sub_out, exist_ok=True)
                    self.log(f"[batch {idx}/{len(files)}] parse & export: {msr_path} -> {sub_out}")

                    # Parse this file (fills MFSTATE maps; parse_msr_general does a reset internally)
                    res = parse_msr_general(str(msr_path), tmp, log=self.log)
                    self.parsed = res  # keep for dataset order/names

                    # Export everything from the current parsed state into the file's subfolder
                    self._export_current_parsed(str(sub_out), formats,
                                                mfx_sel_global=mfx_sel_global,
                                                mbm_sel_global=mbm_sel_global)

                    # Yield to UI so you see progress immediately
                    self.update_idletasks()

                except Exception as e:
                    # Continue to next file on errors, but log them
                    self.log(f"[batch error] {msr_path}: {e}")
                    continue

            self.log("[done] Batch export complete.")
            return

        # ---------- SINGLE FILE MODE ----------
        # Ensure we have a parsed file
        if not self.parsed:
            from pathlib import Path
            msr = ipath if os.path.isfile(ipath) else next((str(p) for p in Path(ipath).glob("*.msr")), None)
            if not msr:
                self.log("no msr file found!")
                return
            self.parsed = parse_msr_general(msr, tmp, log=self.log)

        # Export current parsed datasets into chosen out_dir, honoring per-dataset selection
        self._export_current_parsed(out_dir, formats,
                                    mfx_sel_global=None,
                                    mbm_sel_global=None)

        self.log("[done] Export complete.")


def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
