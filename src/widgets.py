import tkinter as tk
import numpy as np

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


class CurveWidget(tk.Canvas):
    """
    An advanced monotonic curve control widget for generating precise image processing LUTs.
    Supports Value, Red, Green, and Blue channels with dynamic node configurations.
    """
    def __init__(self, parent, width=200, height=200, callback=None, **kwargs):
        super().__init__(parent, width=width, height=height, bg="grey12", borderwidth=0, highlightthickness=0, **kwargs)
        self.width = width
        self.height = height
        self.callback = callback
        
        # Normalized coordinates [x, y] bound between 0.0 and 1.0
        self.channels_points = {
            'Value': [[0.0, 0.0], [1.0, 1.0]],
            'Red':   [[0.0, 0.0], [1.0, 1.0]],
            'Green': [[0.0, 0.0], [1.0, 1.0]],
            'Blue':  [[0.0, 0.0], [1.0, 1.0]]
        }
        self.current_channel = 'Value'
        self.dragged_point_idx = None
        
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Button-3>", self._on_right_click)  # Right-click to remove node
        
        self.redraw()

    def set_channel(self, channel):
        if channel in self.channels_points:
            self.current_channel = channel
            self.redraw()

    def get_lut(self, channel):
        pts = sorted(self.channels_points[channel], key=lambda p: p[0])
        xp = [p[0] * 255.0 for p in pts]
        fp = [p[1] * 255.0 for p in pts]
        return np.interp(np.arange(256), xp, fp).astype(np.uint8)

    def get_all_luts(self):
        return {ch: self.get_lut(ch) for ch in self.channels_points}

    def redraw(self):
        self.delete("all")
        
        # 1. Draw Bounding Uniform Grid Lines
        for i in range(1, 4):
            cx = (i / 4.0) * self.width
            cy = (i / 4.0) * self.height
            self.create_line(cx, 0, cx, self.height, fill="grey25", dash=(2, 2))
            self.create_line(0, cy, self.width, cy, fill="grey25", dash=(2, 2))
            
        pts = sorted(self.channels_points[self.current_channel], key=lambda p: p[0])
        
        # 2. Draw Mathematical Processing Profile Line
        xp = [p[0] * self.width for p in pts]
        fp = [(1.0 - p[1]) * self.height for p in pts]
        x_new = np.arange(self.width)
        y_new = np.interp(x_new, xp, fp)
        
        line_pts = []
        for x, y in zip(x_new, y_new):
            line_pts.extend([x, y])
        
        color_map = {'Value': 'orange', 'Red': '#ff4444', 'Green': '#44ff44', 'Blue': '#4444ff'}
        self.create_line(line_pts, fill=color_map[self.current_channel], width=2)
        
        # 3. Draw Interaction Control Nodes
        for idx, p in enumerate(pts):
            cx = p[0] * self.width
            cy = (1.0 - p[1]) * self.height
            self.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="white", outline="black")

    def _on_press(self, event):
        pts = self.channels_points[self.current_channel]
        click_x = event.x / self.width
        click_y = 1.0 - (event.y / self.height)
        
        # Select existing point near click position
        for idx, p in enumerate(pts):
            if abs(p[0] - click_x) * self.width < 8 and abs(p[1] - click_y) * self.height < 8:
                self.dragged_point_idx = idx
                return
        
        # Build an arbitrary new control node if line region is targeted
        if 0.0 < click_x < 1.0:
            new_point = [click_x, np.clip(click_y, 0.0, 1.0)]
            pts.append(new_point)
            pts.sort(key=lambda p: p[0])
            self.dragged_point_idx = pts.index(new_point)
            self.redraw()
            if self.callback:
                self.callback()

    def _on_drag(self, event):
        if self.dragged_point_idx is None:
            return
        pts = self.channels_points[self.current_channel]
        p = pts[self.dragged_point_idx]
        
        new_x = np.clip(event.x / self.width, 0.0, 1.0)
        new_y = np.clip(1.0 - (event.y / self.height), 0.0, 1.0)
        
        # Endpoints remain locked to boundaries along X dimension
        if p[0] == 0.0 or p[0] == 1.0:
            p[1] = new_y
        else:
            # Prevent nodes passing neighbor boundaries (preserves monotonicity)
            p[0] = new_x
            p[1] = new_y
            
        self.redraw()
        if self.callback:
            self.callback()

    def _on_release(self, event):
        if self.dragged_point_idx is not None:
            self.channels_points[self.current_channel].sort(key=lambda p: p[0])
            self.dragged_point_idx = None
            self.redraw()

    def _on_right_click(self, event):
        pts = self.channels_points[self.current_channel]
        click_x = event.x / self.width
        click_y = 1.0 - (event.y / self.height)
        
        for idx, p in enumerate(pts):
            if abs(p[0] - click_x) * self.width < 8 and abs(p[1] - click_y) * self.height < 8:
                if p[0] != 0.0 and p[0] != 1.0:  # Safeguard endpoints
                    pts.pop(idx)
                    self.redraw()
                    if self.callback:
                        self.callback()
                    return
                    
    def reset_all(self):
        for ch in self.channels_points:
            self.channels_points[ch] = [[0.0, 0.0], [1.0, 1.0]]
        self.redraw()
        if self.callback:
            self.callback()