import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageEnhance
import tifffile
import os
import zarr

PREVIEW_MAX = 512
TILE_PREVIEW_SIZE = 256
TILE_LEVEL_OFFSET = 1

def to_uint8(arr):
    a = arr.astype(np.float32)
    mn, mx = a.min(), a.max()
    if mx > mn:
        a = (a - mn) / (mx - mn) * 255
    else:
        a = np.zeros_like(a)
    return a.astype(np.uint8)

def apply_adjustments(base_rgb, r_gain, g_gain, b_gain, brightness, contrast, saturation, hue_shift=0.0):
    img = base_rgb.astype(np.float32)
    img[:, :, 0] *= r_gain
    img[:, :, 1] *= g_gain
    img[:, :, 2] *= b_gain
    img = np.clip(img, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img, "RGB")
    pil = ImageEnhance.Brightness(pil).enhance(brightness)
    pil = ImageEnhance.Contrast(pil).enhance(contrast)
    pil = ImageEnhance.Color(pil).enhance(saturation)
    if hue_shift != 0.0:
        hsv = np.array(pil.convert("HSV"), dtype=np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + int(hue_shift / 360 * 256)) % 256
        pil = Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")
    return pil

def draw_histogram(canvas, pil_img, w, h):
    canvas.delete("all")
    arr = np.array(pil_img)
    for ch, color in enumerate(["red", "green", "blue"]):
        hist, _ = np.histogram(arr[:, :, ch], bins=64, range=(0, 255))
        hist = np.log1p(hist)
        hist = hist / (hist.max() + 1e-6)
        pts = []
        for i, v in enumerate(hist):
            pts.extend([int(i / 64 * w), int(h - v * h)])
        pts.extend([w, h, 0, h])
        canvas.create_polygon(pts, fill=color, outline="", stipple="gray50")

def draw_hue_profile(canvas, pil_img, w, h):
    import colorsys
    canvas.delete("all")
    rgb = np.array(pil_img).astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mx = np.max(rgb, axis=2)
    mn = np.min(rgb, axis=2)
    delta = mx - mn + 1e-9
    hue = np.zeros_like(mx)
    mask_r = (mx == r)
    mask_g = (mx == g)
    mask_b = (mx == b)
    hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360
    hue[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120) % 360
    hue[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240) % 360
    hue[delta < 0.01] = -1
    for x in range(w):
        rv, gv, bv = colorsys.hsv_to_rgb(x / w, 1.0, 0.8)
        col = "#%02x%02x%02x" % (int(rv*255), int(gv*255), int(bv*255))
        canvas.create_line(x, 0, x, h, fill=col)
    hist, _ = np.histogram(hue[hue >= 0], bins=w, range=(0, 360))
    if hist.max() > 0:
        hist = hist / hist.max()
    pts = [0, h]
    for x, v in enumerate(hist):
        pts.extend([x, int(h - v * h * 0.9)])
    pts.extend([w, h])
    canvas.create_polygon(pts, fill="white", outline="", stipple="gray75")


class TiffColorizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TIFF Colorizer")
        self.tiff_path = None
        self.base_rgb = None
        self._update_job = None
        self._zarr_level = None
        self._build_ui()

    def _build_ui(self):
        top = tk.Frame(self)
        top.pack(fill="x", padx=4, pady=4)
        tk.Button(top, text="Open TIFF", command=self._open_tiff).pack(side="left")
        tk.Button(top, text="Save as New TIFF", command=self._save_tiff).pack(side="left", padx=4)
        tk.Button(top, text="Reset", command=self._reset).pack(side="left")
        self.file_label = tk.Label(top, text="No file loaded")
        self.file_label.pack(side="left", padx=8)

        main = tk.Frame(self)
        main.pack(fill="both", expand=True, padx=4, pady=4)

        left = tk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        self.canvas_frame = tk.Frame(left, width=PREVIEW_MAX, height=PREVIEW_MAX)
        self.canvas_frame.pack();
        self.canvas_frame.pack_propagate(False)

        self.canvas = tk.Canvas(self.canvas_frame, width=PREVIEW_MAX, height=PREVIEW_MAX, bg="blue")
        self.canvas.pack(fill="both", expand=True)

        self.preview = tk.Canvas(self.canvas_frame, width=TILE_PREVIEW_SIZE, height=TILE_PREVIEW_SIZE, bg="black")
        total = TILE_PREVIEW_SIZE
        x = PREVIEW_MAX - total - 16
        y = PREVIEW_MAX - total - 16
        self.preview.place(x=x, y=y)
        self.preview.pack_propagate(False)

        btn_cfg = dict(text="", width=2, relief="flat", bd=0,
                       bg="#555555", fg="#555555", activebackground="#888888",
                       font=("TkDefaultFont", 7))
        self.preview_btn_up = tk.Button(self.preview, text="^", cnf=btn_cfg, width=1, height=1, bg="gray", command=lambda: self._move_preview_tile(0, -TILE_PREVIEW_SIZE))
        self.preview_btn_up.pack(side="top")
        self.preview_btn_down = tk.Button(self.preview, text="~", cnf=btn_cfg, width=1, height=1, bg="gray", command=lambda: self._move_preview_tile(0, TILE_PREVIEW_SIZE))
        self.preview_btn_down.pack(side="bottom")
        self.preview_btn_left = tk.Button(self.preview, text="<", cnf=btn_cfg, width=1, height=1, bg="gray", command=lambda: self._move_preview_tile(-TILE_PREVIEW_SIZE, 0))
        self.preview_btn_left.pack(side="left")
        self.preview_btn_right = tk.Button(self.preview, text=">", cnf=btn_cfg, width=1, height=1, bg="gray", command=lambda: self._move_preview_tile(TILE_PREVIEW_SIZE, 0))
        self.preview_btn_right.pack(side="right")

        tk.Label(left, text="Histogram").pack(anchor="w")
        self.hist_canvas = tk.Canvas(left, width=PREVIEW_MAX, height=80, bg="white")
        self.hist_canvas.pack(fill="x")

        tk.Label(left, text="Hue Profile").pack(anchor="w")
        self.hue_canvas = tk.Canvas(left, width=PREVIEW_MAX, height=50, bg="white")
        self.hue_canvas.pack(fill="x")

        right = tk.Frame(main)
        right.pack(side="right", fill="y", padx=8)

        def slider(label, from_, to, init, res=0.01):
            tk.Label(right, text=label, anchor="w").pack(fill="x")
            var = tk.DoubleVar(value=init)
            tk.Scale(right, variable=var, from_=from_, to=to, resolution=res,
                     orient="horizontal", length=200,
                     command=lambda _: self._schedule_update()).pack()
            return var
        
        tk.Label(right, text="Initial").pack(anchor="w", pady=(8, 0))
        self.invert = tk.BooleanVar(value=False)
        tk.Button(right, text="Invert")

        tk.Label(right, text="RGB Gain").pack(anchor="w", pady=(8, 0))
        self.r_gain     = slider("R", 0.0, 3.0, 1.0)
        self.g_gain     = slider("G", 0.0, 3.0, 1.0)
        self.b_gain     = slider("B", 0.0, 3.0, 1.0)

        tk.Label(right, text="Image").pack(anchor="w", pady=(8, 0))
        self.brightness = slider("Brightness", 0.1, 3.0, 1.0)
        self.contrast   = slider("Contrast",   0.1, 3.0, 1.0)
        self.saturation = slider("Saturation", 0.0, 3.0, 1.0)
        
        tk.Label(right, text="Color").pack(anchor="w", pady=(8, 0))
        self.hue_shift = slider("Hue Shift", -180, 180, 0.0, res=1.0)

    def _open_tiff(self):
        path = filedialog.askopenfilename(
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")])
        if not path:
            return
        try:
            with tifffile.TiffFile(path) as tif:
                series = tif.series[0] if tif.series else None
                if series and len(series.levels) > 1:
                    chosen_level = series.levels[-1]
                    for level in reversed(series.levels):
                        shape = level.shape
                        h = shape[-3] if len(shape) >= 3 else shape[0]
                        w = shape[-2] if len(shape) >= 3 else shape[1]
                        if max(h, w) >= PREVIEW_MAX:
                            chosen_level = level
                            break
                    arr = chosen_level.pages[0].asarray()
                    s0 = series.levels[0].shape
                    base_h = s0[-3] if len(s0) >= 3 else s0[0]
                    base_w = s0[-2] if len(s0) >= 3 else s0[1]
                else:
                    page = (series.pages[0] if series else tif.pages[0])
                    arr = page.asarray()
                    base_h, base_w = arr.shape[0], arr.shape[1]

            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.shape[2] > 3:
                arr = arr[:, :, :3]
            arr = to_uint8(arr)

            store = tifffile.imread(path, aszarr=True)
            z = zarr.open(store, mode='r')
            lvl_idx = min(TILE_LEVEL_OFFSET, len(z) - 1) if hasattr(z, '__len__') else 0
            self._zarr_level = z[str(lvl_idx)] if hasattr(z, '__len__') else z
            full_h, full_w = self._zarr_level.shape[0], self._zarr_level.shape[1]
            
            tile_size = TILE_PREVIEW_SIZE
            cx, cy = full_w // 2, full_h // 2
            x0 = max(0, cx - tile_size // 2)
            y0 = max(0, cy - tile_size // 2)
            x1 = min(full_w, x0 + tile_size)
            y1 = min(full_h, y0 + tile_size)
            
            # tile by tile loading improvement
            tile = self._zarr_level[y0:y1, x0:x1]
            
            if tile.ndim == 2:
                tile = np.stack([tile, tile, tile], axis=-1)
            elif tile.shape[2] > 3:
                tile = tile[:, :, :3]
            self.preview_base_rgb = to_uint8(tile)
            self.preview_position = (x0, y0, x1, y1)   # pixels in full-res coords

            self.base_rgb = arr
            self.tiff_path = path
            self.file_label.config(
                text=f"{os.path.basename(path)}  preview {arr.shape[1]}x{arr.shape[0]}  full {base_w}x{base_h}")
            self._schedule_update()
        except Exception as e:
            messagebox.showerror("Error", f"Could not open TIFF:\n{e}")

    def _save_tiff(self):
        if self.base_rgb is None:
            messagebox.showwarning("No image", "Open a TIFF first.")
            return
        out_path = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=[("TIFF files", "*.tif *.tiff")])
        if not out_path:
            return

        r_gain     = self.r_gain.get()
        g_gain     = self.g_gain.get()
        b_gain     = self.b_gain.get()
        brightness = self.brightness.get()
        contrast   = self.contrast.get()
        saturation = self.saturation.get()

        try:
            messagebox.showinfo("Saving", "Reading full base level — may take a moment.\nClick OK to proceed.")
            with tifffile.TiffFile(self.tiff_path) as tif:
                series = tif.series[0] if tif.series else None
                base_page = (series.levels[0].pages[0] if series else tif.pages[0])
                base_arr = base_page.asarray()
                orig_dtype = base_arr.dtype
                other_pages = [p.asarray() for p in tif.pages[1:]]

            if base_arr.ndim == 2:
                base_arr = np.stack([base_arr]*3, axis=-1)
            elif base_arr.shape[2] > 3:
                base_arr = base_arr[:, :, :3]

            adj_pil = apply_adjustments(to_uint8(base_arr), r_gain, g_gain, b_gain,
                            brightness, contrast, saturation,
                            self.hue_shift.get())
            adj_arr = np.array(adj_pil)

            if np.issubdtype(orig_dtype, np.integer):
                max_val = np.iinfo(orig_dtype).max
                save_base = (adj_arr.astype(np.float32) / 255.0 * max_val).astype(orig_dtype)
            else:
                save_base = adj_arr.astype(orig_dtype)

            with tifffile.TiffWriter(out_path, bigtiff=True) as tw:
                tw.write(save_base)
                for page in other_pages:
                    tw.write(page)

            messagebox.showinfo("Saved", f"Saved to:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save:\n{e}")

    def _get_adjusted_pil(self):
        return apply_adjustments(
            self.base_rgb,
            self.r_gain.get(), self.g_gain.get(), self.b_gain.get(),
            self.brightness.get(), self.contrast.get(), self.saturation.get(),
            self.hue_shift.get())

    def _schedule_update(self):
        if self._update_job:
            self.after_cancel(self._update_job)
        self._update_job = self.after(60, self._update_preview)

    def _update_preview(self):
        if self.base_rgb is None:
            return
        pil = self._get_adjusted_pil()
        w, h = pil.size
        scale = min(PREVIEW_MAX / w, PREVIEW_MAX / h, 1.0)
        pw, ph = max(1, int(w * scale)), max(1, int(h * scale))
        thumb = pil.resize((pw, ph), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(thumb)
        self.canvas.config(width=pw, height=ph)
        self.canvas.delete("all")
        self.canvas.create_image(0, PREVIEW_MAX / 2, anchor="w", image=self._tk_img)
        cw = self.hist_canvas.winfo_width() or PREVIEW_MAX
        huw = self.hue_canvas.winfo_width() or PREVIEW_MAX
        draw_histogram(self.hist_canvas, thumb, cw, 80)
        draw_hue_profile(self.hue_canvas, thumb, huw, 50)

        if hasattr(self, "preview_base_rgb") and self.preview_base_rgb is not None:
            tile_pil = apply_adjustments(
                self.preview_base_rgb,
                self.r_gain.get(), self.g_gain.get(), self.b_gain.get(),
                self.brightness.get(), self.contrast.get(), self.saturation.get(),
                self.hue_shift.get())
            pw2 = self.preview.winfo_width()  or TILE_PREVIEW_SIZE
            ph2 = self.preview.winfo_height() or TILE_PREVIEW_SIZE
            tile_pil = tile_pil.resize((pw2, ph2), Image.NEAREST)
            self._preview_tk = ImageTk.PhotoImage(tile_pil)
            self.preview.delete("all")
            self.preview.create_image(0, 0, anchor="nw", image=self._preview_tk)

    def _move_preview_tile(self, dx, dy):
        if self._zarr_level is None or not hasattr(self, "preview_position"):
            return
        z = self._zarr_level
        full_h, full_w = z.shape[0], z.shape[1]
        x0, y0, x1, y1 = self.preview_position
        tile_w = x1 - x0
        tile_h = y1 - y0

        nx0 = max(0, min(x0 + dx, full_w - tile_w))
        ny0 = max(0, min(y0 + dy, full_h - tile_h))
        nx1 = nx0 + tile_w
        ny1 = ny0 + tile_h

        tile = np.array(z[ny0:ny1, nx0:nx1])
        if tile.ndim == 2:
            tile = np.stack([tile, tile, tile], axis=-1)
        elif tile.shape[2] > 3:
            tile = tile[:, :, :3]
        self.preview_base_rgb = to_uint8(tile)
        self.preview_position = (nx0, ny0, nx1, ny1)
        self._schedule_update()

    def _reset(self):
        for var in (self.r_gain, self.g_gain, self.b_gain):
            var.set(1.0)
        for var in (self.brightness, self.contrast, self.saturation):
            var.set(1.0)
        self.hue_shift.set(0.0)
        self._schedule_update()


if __name__ == "__main__":
    TiffColorizer().mainloop()