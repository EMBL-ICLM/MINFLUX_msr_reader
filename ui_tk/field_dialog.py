# ui_tk/field_dialog.py
import tkinter as tk
from tkinter import ttk
from functools import partial
from typing import Dict, List, Optional
from functools import partial

class FieldDialog(tk.Toplevel):
    """
    Modal dialog to select datasets and fields, displayed horizontally.

    Input:
      datasets: list of dicts:
        {
          "key": str,               # dataset key (display name or DID)
          "name": str,              # display name shown in UI
          "mfx_fields": List[str],  # e.g., ['vld','fnl',...]
          "mbm_fields": List[str],  # e.g., ['gri','xyz','tim','str']
        }
      prechecked (optional):
        {
          <key>: {
            "mfx": set([...]) or None (=> all),
            "mbm": set([...]) or None (=> all),
            "checked": bool (dataset on/off)
          }, ...
        }

    Output (on OK): dict with same shape as prechecked.
    """

    def __init__(self, master, datasets: List[dict], prechecked: Optional[Dict[str, dict]] = None):
        super().__init__(master)
        self.title("Datasets / Fields included…")
        self.transient(master)
        self.grab_set()

        self.datasets = datasets
        self.result = None
        self.vars: Dict[str, dict] = {}  # key -> dict of tk vars

        # --- main container using grid so datasets can sit side-by-side ---
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        # helper function
        def _toggle_all(children_vars: dict, master_var: tk.BooleanVar):
            val = master_var.get()
            for vv in children_vars.values():
                vv.set(val)


        # One column per dataset
        num = max(1, len(datasets))
        for c in range(num):
            outer.grid_columnconfigure(c, weight=1)

        # Build one panel per dataset (horizontally)
        for col, ds in enumerate(datasets):
            key = ds["key"]
            name = ds["name"]
            mfx_fields = ds.get("mfx_fields", [])
            mbm_fields = ds.get("mbm_fields", [])

            prev = prechecked.get(key, {}) if prechecked else {}
            ds_checked_default = prev.get("checked", True)

            # Panel frame
            panel = ttk.LabelFrame(outer, text=name)
            panel.grid(row=0, column=col, sticky="nsew", padx=8, pady=6)
            panel.grid_rowconfigure(99, weight=1)  # spacer

            # Dataset include checkbox
            v_ds = tk.BooleanVar(value=bool(ds_checked_default))
            ttk.Checkbutton(panel, text="Include dataset", variable=v_ds).grid(
                row=0, column=0, sticky="w", padx=8, pady=(6, 4)
            )

            # --- mfx group ---
            v_mfx_all = tk.BooleanVar(value=(prev.get("mfx") is None if prechecked else True))
            mfx_group = ttk.LabelFrame(panel, text="mfx fields")
            mfx_group.grid(row=1, column=0, sticky="nsew", padx=8, pady=(2, 6))
            # Dict of child vars
            v_mfx: Dict[str, tk.BooleanVar] = {}
            pre_mfx = set(prev.get("mfx") or []) if prechecked else set()

            # All mfx toggle (command ensures per-dataset closure)
            def _toggle_all_mfx(k: str):
                val = v_mfx_all.get()
                for vv in v_mfx.values():
                    vv.set(val)

            ttk.Checkbutton(
                mfx_group,
                text="All mfx fields",
                variable=v_mfx_all,
                command=partial(_toggle_all, v_mfx, v_mfx_all)  # <-- bind concrete vars
            ).pack(anchor="w", padx=8, pady=(4, 2))

            # Child mfx field checkboxes
            # If "all" selected (None), start all True; else use selection set
            for fname in mfx_fields:
                init = True if (prev.get("mfx") is None if prechecked else True) else (fname in pre_mfx)
                v = tk.BooleanVar(value=init)
                ttk.Checkbutton(mfx_group, text=fname, variable=v).pack(anchor="w", padx=20, pady=0)
                v_mfx[fname] = v

            # --- mbm group ---
            v_mbm_all = tk.BooleanVar(value=(prev.get("mbm") is None if prechecked else True))
            mbm_group = ttk.LabelFrame(panel, text="mbm fields")
            mbm_group.grid(row=2, column=0, sticky="nsew", padx=8, pady=(2, 6))
            v_mbm: Dict[str, tk.BooleanVar] = {}
            pre_mbm = set(prev.get("mbm") or []) if prechecked else set()

            def _toggle_all_mbm(k: str):
                val = v_mbm_all.get()
                for vv in v_mbm.values():
                    vv.set(val)

            ttk.Checkbutton(
                mbm_group,
                text="All mbm fields",
                variable=v_mbm_all,
                command=partial(_toggle_all, v_mbm, v_mbm_all)  # <-- bind concrete vars
            ).pack(anchor="w", padx=8, pady=(4, 2))

            for fname in mbm_fields:
                init = True if (prev.get("mbm") is None if prechecked else True) else (fname in pre_mbm)
                v = tk.BooleanVar(value=init)
                ttk.Checkbutton(mbm_group, text=fname, variable=v).pack(anchor="w", padx=20, pady=0)
                v_mbm[fname] = v

            # Store all vars for this dataset key
            self.vars[key] = {
                "ds": v_ds,
                "mfx_all": v_mfx_all, "mfx": v_mfx,
                "mbm_all": v_mbm_all, "mbm": v_mbm,
            }

        # Buttons row
        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(btns, text="OK", command=self.on_ok).pack(side="right", padx=6)
        ttk.Button(btns, text="Cancel", command=self.on_cancel).pack(side="right")

        # Sizing
        self.update_idletasks()
        # Wider if two datasets, else compact
        min_w = 360 if len(datasets) >= 2 else 200
        self.minsize(min_w, 420)
        self.wait_window(self)

    def on_ok(self):
        out = {}
        for key, vv in self.vars.items():
            checked = vv["ds"].get()

            # mfx selection: None => all, else explicit set of checked names
            if vv["mfx_all"].get():
                mfx_sel = None
            else:
                mfx_sel = {k for k, v in vv["mfx"].items() if v.get()}

            # mbm selection
            if vv["mbm_all"].get():
                mbm_sel = None
            else:
                mbm_sel = {k for k, v in vv["mbm"].items() if v.get()}

            out[key] = {"checked": checked, "mfx": mfx_sel, "mbm": mbm_sel}

        self.result = out
        self.destroy()

    def on_cancel(self):
        self.result = None
        self.destroy()
