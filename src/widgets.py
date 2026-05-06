import tkinter as tk


class NumVar:
    """A DoubleVar-like wrapper backed by a tk.StringVar + Entry widget."""

    def __init__(self, master, init, from_, to, callback):
        self._from = from_
        self._to = to
        self._callback = callback
        self._sv = tk.StringVar(master, value=str(init))
        self._sv.trace_add("write", self._on_write)

    def get(self):
        try:
            return float(self._sv.get())
        except ValueError:
            return 0.0

    def set(self, value):
        self._sv.set(str(value))

    def _on_write(self, *_):
        self._callback()

    def make_entry(self, parent, width=8):
        vcmd = (parent.register(self._validate), "%P")
        e = tk.Entry(parent, textvariable=self._sv, width=width,
                     validate="key", validatecommand=vcmd)
        return e

    def _validate(self, new_val):
        if new_val in ("", "-", ".", "-."):
            return True
        try:
            float(new_val)
            return True
        except ValueError:
            return False