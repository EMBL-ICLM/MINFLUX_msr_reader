import os
from pathlib import Path
from typing import Optional, Dict, Any, List

# NEW: optional DnD
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _DND = True
    BaseTk = TkinterDnD.Tk
except Exception:
    _DND = False
    BaseTk = None  # set after tk import
    
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

if BaseTk is None:  # fallback when tkinterdnd2 not available
    BaseTk = tk.Tk
    
from minflux_msr import (
    slug, pick_one_msr, parse_msr_to_tree, process_file,
    export_selected_fields, make_export_base
)
from .field_dialog import FieldSelectDialog


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


class App(BaseTk):
    def __init__(self):
        super().__init__()
        self.title("MINFLUX .msr Exporter (SpecPy + Zarr)")
        self.geometry("550x800")

        # Inputs
        self.mode = tk.StringVar(value="file")
        self.input_path = tk.StringVar(value="")
        self.tmp_dir = tk.StringVar(value=r"C:\data\temp")
        self._fullpath_by_node: Dict[str, str] = {}      # node_id -> zarr path ('' for dataset)
        self._datasetnode_info: Dict[str, Dict[str, Any]] = {}  # ds node -> {'did':..,'zroot':..}


        # Export options
        self.export_mode = tk.StringVar(value=EXPORT_MODES[2])
        self.out_mat = tk.BooleanVar(value=True)
        self.out_npy = tk.BooleanVar(value=False)
        self.out_csv = tk.BooleanVar(value=False)

        # Parsed cache & selections
        self.parsed: Dict[str, List[Dict[str, Any]]] = {}   # { msr_file: [ {idx,did,display_name,zroot,fields} ] }
        self.selected_fields: Dict[str, set] = {}           # { did: set(paths) }

        # Lazy preview
        self._nodeinfo: Dict[str, Dict[str, Any]] = {}      # node_id -> {zroot, path, preview}
        self.PREVIEW_N = 200

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        row = ttk.Frame(self); row.pack(fill="x", **pad)
        ttk.Label(row, text="Input type:").pack(side="left")
        ttk.Radiobutton(row, text="Single .msr file", variable=self.mode, value="file").pack(side="left", padx=8)
        ttk.Radiobutton(row, text="Folder (batch)", variable=self.mode, value="folder").pack(side="left", padx=8)

        # MSR file or folder row
        row = ttk.Frame(self); row.pack(fill="x", **pad)
        ttk.Label(row, text="MSR file or folder:").pack(side="left")
        self.input_entry = ttk.Entry(row, textvariable=self.input_path)  # keep handle
        self.input_entry.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row, text="Browse…", command=self.browse_input).pack(side="left")

        row = ttk.Frame(self); row.pack(fill="x", **pad)
        ttk.Label(row, text="Temp directory (Zarr + exports):").pack(side="left")
        ttk.Entry(row, textvariable=self.tmp_dir).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row, text="Browse…", command=self.browse_tmp).pack(side="left")
        # register drag & drop
        if _DND:
            try:
                # accept drops on the entry and the whole window
                self.input_entry.drop_target_register(DND_FILES)
                self.input_entry.dnd_bind("<<Drop>>", self.on_drop_input)
                self.drop_target_register(DND_FILES)
                self.dnd_bind("<<Drop>>", self.on_drop_input)
            except Exception:
                pass

        row = ttk.Frame(self); row.pack(fill="x", **pad)
        ttk.Button(row, text="Parse MSR file", command=self.on_parse).pack(side="left")
        ttk.Button(row, text="Fields included…", command=self.on_fields_dialog).pack(side="left", padx=8)
        ttk.Label(row, text="(Select a dataset in the tree, then choose its fields)").pack(side="left", padx=12)

        tvf = ttk.LabelFrame(self, text="Parsed datasets & fields (Zarr)")
        tvf.pack(fill="both", expand=True, **pad)
        self.tree = ttk.Treeview(tvf, columns=("kind","shape","dtype"), show="tree headings", height=14)
        self.tree.heading("#0", text="Path")
        self.tree.heading("kind", text="Kind")
        self.tree.heading("shape", text="Shape")
        self.tree.heading("dtype", text="DType")
        self.tree.column("#0", width=420); self.tree.column("kind", width=90)
        self.tree.column("shape", width=180); self.tree.column("dtype", width=160)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewOpen>>", self.on_tree_open)
        self._attach_context_menu()
        
        row = ttk.Frame(self); row.pack(fill="x", **pad)
        ttk.Label(row, text="Export mode:").pack(side="left")
        ttk.Combobox(row, textvariable=self.export_mode, values=EXPORT_MODES, state="readonly", width=28).pack(side="left", padx=8)

        opt = ttk.LabelFrame(self, text="Export formats"); opt.pack(fill="x", **pad)
        ttk.Checkbutton(opt, text="Export .mat (SciPy)", variable=self.out_mat).pack(side="left", padx=10, pady=6)
        ttk.Checkbutton(opt, text="Export .npy / .npz", variable=self.out_npy).pack(side="left", padx=10, pady=6)
        ttk.Checkbutton(opt, text="Export .csv", variable=self.out_csv).pack(side="left", padx=10, pady=6)

        logf = ttk.LabelFrame(self, text="Log"); logf.pack(fill="both", expand=False, **pad)
        self.logbox = ScrolledText(logf, height=8, wrap="word", font=("Consolas", 10)); self.logbox.pack(fill="both", expand=True)

        row = ttk.Frame(self); row.pack(fill="x", **pad)
        ttk.Button(row, text="OK", command=self.on_ok).pack(side="right")
        ttk.Button(row, text="Cancel", command=self.destroy).pack(side="right", padx=8)


    # ---------- helpers ----------

    # ---------- DnD helpers ----------
    def _parse_dnd_list(self, data: str) -> List[str]:
        """Parse tkdnd path list like '{C:\\a b}\\nC:\\c' into ['C:\\a b','C:\\c']"""
        if not data:
            return []
        out, token, brace = [], "", False
        for ch in data:
            if ch == "{":
                brace, token = True, ""
                continue
            if ch == "}" and brace:
                brace = False
                out.append(token)
                token = ""
                continue
            if (ch in (" ", "\n", "\r", "\t")) and not brace:
                if token:
                    out.append(token)
                    token = ""
                continue
            token += ch
        if token:
            out.append(token)
        return out

    def on_drop_input(self, event):
        paths = self._parse_dnd_list(getattr(event, "data", "") or "")
        if not paths:
            return
        p = paths[0]
        self.input_path.set(p)
        # optional nicety: switch mode to folder/file
        try:
            if os.path.isdir(p):
                self.mode.set("folder")
            elif os.path.isfile(p):
                self.mode.set("file")
        except Exception:
            pass
        self.log(f"[dnd] set input to: {p}")

    # ---------- Right-click (context menu) ----------
    def _attach_context_menu(self):
        self._ctx_node = None
        self._ctx_menu = tk.Menu(self, tearoff=0)
        self._ctx_menu.add_command(label="Preview (first 200)", command=self._ctx_preview)
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
        """Preview first 200 entries of selected field/array in a popup."""
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

        N = min(200, arr.shape[0] if hasattr(arr, "shape") and arr.ndim >= 1 else 0)
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


    def browse_input(self):
        if self.mode.get() == "file":
            p = filedialog.askopenfilename(title="Choose an .msr file", filetypes=[("Imspector .msr", "*.msr"), ("All files", "*.*")])
        else:
            p = filedialog.askdirectory(title="Choose a folder of .msr files")
        if p: self.input_path.set(p)

    def browse_tmp(self):
        p = filedialog.askdirectory(title="Choose temp directory")
        if p: self.tmp_dir.set(p)

    def log(self, msg: str):
        self.logbox.insert("end", msg + "\n"); self.logbox.see("end"); self.logbox.update_idletasks()
        print(msg)

    # ---------- parse & tree ----------

    def on_parse(self):
        ipath = self.input_path.get().strip(); tmp = self.tmp_dir.get().strip()

        # clear previous
        self.tree.delete(*self.tree.get_children())
        self.parsed.clear(); self.selected_fields.clear(); self._nodeinfo.clear()

        msr = pick_one_msr(ipath) if ipath and os.path.exists(ipath) else None
        if not msr or not tmp:
            self.log("no msr file found!"); return
        os.makedirs(tmp, exist_ok=True)

        self.log(f"[parse] using: {msr}")
        recs = parse_msr_to_tree(msr, tmp, log=self.log)
        self.parsed[msr] = recs
        self._fill_tree_for_file(msr, recs)

    def _logical_shape_str(self, array_entry: Dict[str, Any], sub: Optional[Dict[str, Any]]) -> str:
        """
        Display '1×N' for scalar fields, 'k×N' for vector sub-fields where k is sub shape[0].
        """
        N = array_entry.get("length", 0)
        if not sub:
            sh = array_entry.get("shape")
            return str(sh) if sh is not None else ""
        s = sub.get("shape") or ()
        k = 1 if not s else int(s[0]) if isinstance(s, (list, tuple)) else 1
        return f"{k}×{N}"

    def _fill_tree_for_file(self, msr_file: str, ds_list: List[Dict[str, Any]]):
        root_id = self.tree.insert("", "end",
                                text=os.path.basename(msr_file),
                                values=("", "", ""), open=True)

        for ds in ds_list:
            did = ds["did"]
            ds_label = ds.get("display_name") or ds.get("name") or did
            ds_id = self.tree.insert(root_id, "end",
                                    text=ds_label,
                                    values=("dataset", "", did),
                                    open=True)
            # NEW: remember dataset info
            self._datasetnode_info[ds_id] = {"did": did, "zroot": ds["zroot"]}
            self._fullpath_by_node[ds_id] = ""  # dataset root

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
                    # NEW: remember full zarr path for this node
                    self._fullpath_by_node[node_id] = cur

                    if is_last and kind == "array":
                        # structured subfields (already present in your rolled-back build)
                        for sub in fld.get("dtype_fields", []) or []:
                            sub_node = self.tree.insert(node_id, "end",
                                                        text=sub["name"],
                                                        values=("field",
                                                                self._logical_shape_str(fld, sub),
                                                                sub.get("dtype", "")))
                            self._fullpath_by_node[sub_node] = f"{path}/{sub['name']}"
                        # lazy preview child
                        preview_id = self.tree.insert(node_id, "end",
                                                    text=f"preview (first {self.PREVIEW_N} rows)",
                                                    values=("preview", "", ""))
                        self._nodeinfo[preview_id] = {"zroot": ds["zroot"], "path": path, "preview": True}


    def on_tree_open(self, event=None):
        node = self.tree.focus()
        info = self._nodeinfo.get(node)
        if not info or not info.get("preview"):
            return

        self._nodeinfo.pop(node, None)
        zroot = info["zroot"]; path = info["path"]
        try:
            import zarr, numpy as np
            arch = zarr.open(zroot, mode="r")
            arr = arch[path]
            N_total = int(arr.shape[0]) if arr.shape else 0
            n = min(self.PREVIEW_N, N_total)

            # Estimate 'm' from 'itr' if present
            m_txt = ""
            try:
                if arr.dtype.names and ("itr" in arr.dtype.names):
                    sample = min(max(self.PREVIEW_N * 50, n), N_total)
                    its = np.asarray(arr["itr"][:sample])
                    m_est = int(np.unique(its).size)
                    m_txt = f", m≈{m_est}"
            except Exception:
                pass

            self.tree.item(node, text=f"preview (first {n} rows; N={N_total}{m_txt})")
            for i in range(n):
                row = arr[i]
                try:
                    names = getattr(arr.dtype, "names", None)
                    if names:
                        head = []
                        for k in names[:6]:
                            v = row[k]; v = np.asarray(v)
                            if v.ndim == 0: head.append(f"{k}={v.item()}")
                            elif v.size <= 3: head.append(f"{k}={v.tolist()}")
                            else: head.append(f"{k}=[...]")
                        label = f"[{i}] " + ", ".join(head)
                    else:
                        label = f"[{i}] {np.asarray(row).tolist()}"
                except Exception:
                    label = f"[{i}] {str(row)[:120]}..."
                self.tree.insert(node, "end", text=label, values=("", "", ""))
        except Exception as e:
            self.tree.insert(node, "end", text=f"(preview error: {e})", values=("", "", ""))

    # ---------- field selection & export ----------

    def on_fields_dialog(self):
        """Open dialog for the SELECTED DATASET ONLY; list its array + sub-field paths with checkboxes."""
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Select a dataset", "Click a dataset row in the tree first.")
            return

        # bubble up to dataset node (child under file root)
        node = sel[0]
        parent = self.tree.parent(node)
        while parent and self.tree.parent(parent):
            node = parent
            parent = self.tree.parent(parent)

        did = self.tree.set(node, "dtype")  # we stored DID here
        if not did:
            messagebox.showinfo("Select a dataset", "Please select a dataset row (with arrays underneath).")
            return

        # lookup dataset record
        rec = None
        for _, ds_list in self.parsed.items():
            for d in ds_list:
                if d["did"] == did:
                    rec = d; break
            if rec: break
        if not rec:
            messagebox.showerror("Not parsed", "Dataset not found. Parse again.")
            return

        # Build selectable field paths (arrays + structured sub-fields)
        selectable: List[Dict[str, Any]] = []
        for f in (rec.get("fields") or []):
            if f.get("kind") != "array":
                continue
            subs = f.get("dtype_fields") or []
            if subs:
                for sub in subs:
                    selectable.append({
                        "path": f"{f['path']}/{sub['name']}",
                        "shape": sub.get("shape") or (),
                        "dtype": sub.get("dtype", "")
                    })
            else:
                selectable.append({
                    "path": f["path"],
                    "shape": f.get("shape") or (),
                    "dtype": f.get("dtype", "")
                })

        if not selectable:
            messagebox.showwarning("No fields", "No array fields found under this dataset.")
            return

        pre = self.selected_fields.get(did, set())
        dlg = FieldSelectDialog(self, did, selectable, prechecked=pre)
        self.wait_window(dlg)
        if dlg.result is not None:
            self.selected_fields[did] = set(dlg.result)
            self.log(f"[select] did={did} fields: {sorted(self.selected_fields[did])}")

    def on_ok(self):
        ipath = self.input_path.get().strip(); tmp = self.tmp_dir.get().strip()
        if not ipath or not os.path.exists(ipath) or not tmp:
            messagebox.showerror("Missing paths", "Please set an input .msr or folder and temp directory.")
            return
        os.makedirs(tmp, exist_ok=True)

        exts = []
        if self.out_mat.get(): exts.append("mat")
        if self.out_npy.get(): exts.append("npy")
        if self.out_csv.get(): exts.append("csv")
        if not exts:
            messagebox.showerror("No formats", "Select at least one export format.")
            return

        do_mfx, do_mbm, combine = parse_export_mode(self.export_mode.get())
        self.log("=== START ===")

        # If any dataset has selections, export only those (paths may be 'mfx/vld', etc.)
        any_selected = any(self.selected_fields.get(d.get("did")) for recs in self.parsed.values() for d in recs)
        if any_selected:
            for recs in self.parsed.values():
                for d in recs:
                    did = d["did"]; zroot = d["zroot"]; idx = d.get("idx", 1)
                    paths = sorted(self.selected_fields.get(did, set()))
                    if not paths:
                        continue
                    msr_tag = Path(zroot).parents[2].name.replace("msr_", "")
                    export_root = os.path.join(tmp, "exports", f"msr_{slug(msr_tag)}")
                    os.makedirs(export_root, exist_ok=True)
                    base = make_export_base(export_root, did, idx)
                    export_selected_fields(zroot, did, paths, base, exts, log=self.log)
        else:
            # classic flow
            msr = pick_one_msr(ipath)
            if not msr:
                self.log("no msr file found!")
                return
            process_file(msr, tmp, do_mfx, do_mbm, combine, exts, log=self.log)

        self.log("=== DONE ===")
        messagebox.showinfo("Finished", "Export completed. Check the log and export folders.")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
