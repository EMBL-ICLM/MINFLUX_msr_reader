import tkinter as tk
from tkinter import ttk

import numpy as np


class PlotWindow(tk.Toplevel):
    def __init__(self, master, data, title: str = "Plot"):
        super().__init__(master)
        self.title(title)
        self.geometry("900x650")

        self._raw = np.asarray(data)
        self._series = self._to_series_matrix(self._raw)
        self.mode = tk.StringVar(value="line")

        self._build_ui()
        self._draw()

    @staticmethod
    def _to_series_matrix(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        if a.ndim == 1:
            return a.reshape(-1, 1)
        if a.ndim > 2:
            return a.reshape(a.shape[0], -1)

        r, c = a.shape
        if r in (2, 3) and c not in (2, 3):
            return a.T
        if c in (2, 3):
            return a
        return a

    def _build_ui(self):
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        row = ttk.Frame(self)
        row.pack(fill="x", padx=8, pady=8)
        ttk.Label(row, text="Mode:").pack(side="left")
        ttk.Radiobutton(row, text="Line", variable=self.mode, value="line", command=self._draw).pack(side="left", padx=6)
        ttk.Radiobutton(row, text="Scatter", variable=self.mode, value="scatter", command=self._draw).pack(side="left", padx=6)
        ttk.Radiobutton(row, text="Histogram", variable=self.mode, value="hist", command=self._draw).pack(side="left", padx=6)

        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

    def _draw(self):
        self.fig.clf()
        m = self._series
        cols = m.shape[1] if m.ndim == 2 else 1
        mode = self.mode.get()

        if mode == "scatter" and cols >= 3:
            ax = self.fig.add_subplot(111, projection="3d")
            ax.scatter(m[:, 0], m[:, 1], m[:, 2], s=8)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
        else:
            ax = self.fig.add_subplot(111)
            if mode == "line":
                x = np.arange(m.shape[0])
                for i in range(cols):
                    ax.plot(x, m[:, i], label=f"ch{i}")
                if cols > 1:
                    ax.legend()
                ax.set_xlabel("index")
                ax.set_ylabel("value")
            elif mode == "scatter":
                if cols >= 2:
                    ax.scatter(m[:, 0], m[:, 1], s=8)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                else:
                    ax.scatter(np.arange(m.shape[0]), m[:, 0], s=8)
                    ax.set_xlabel("index")
                    ax.set_ylabel("value")
            else:  # hist
                for i in range(cols):
                    ax.hist(m[:, i], bins=50, alpha=0.55, label=f"ch{i}")
                if cols > 1:
                    ax.legend()
                ax.set_xlabel("value")
                ax.set_ylabel("count")

        self.fig.tight_layout()
        self.canvas.draw_idle()
