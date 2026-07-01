import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import tifffile
import zarr
from PIL import Image, ImageTk

from utils import apply_adjustments, draw_histogram, draw_hue_profile, to_uint8
from widgets import NumVar, CurveWidget

PREVIEW_MAX = 512
TILE_PREVIEW_SIZE = 256
TILE_LEVEL_OFFSET = 0

class TiffColorizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TIFF Colorizer")
        self.tiff_path = None
        self.base_rgb = None
        self._update_job = None
        self._zarr_level = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        top = tk.Frame(self)
        top.pack(fill="x", padx=4, pady=4)
        tk.Button(top, text="Open TIFF", command=self._open_tiff).pack(side="left")
        tk.Button(top, text="Save as New TIFF", command=self._save_tiff).pack(side="left", padx=4)
        tk.Button(top, text="Reset", command=self._reset).pack(side="left")
        self.file_label = tk.Label(top, text="No file loaded")
        self.file_label.pack(side="left", padx=8)

        main = tk.Frame(self)
        main.pack(fill="both", expand=True, padx=8, pady=8)

        self._build_left_panel(main)
        self._build_middle_panel(main)
        self._build_right_panel(main)

    def _build_left_panel(self, parent):
        left = tk.Frame(parent)
        left.pack(side="left", fill="y", expand=False, padx=(0, 8))

        self._build_tile_preview(left)

        tk.Label(left, text="Histogram", font=("TkDefaultFont", 9, "bold")).pack(anchor="w", side="bottom", pady=(4, 0))
        self.hist_canvas = tk.Canvas(left, width=PREVIEW_MAX, height=80, bg="#121212")
        self.hist_canvas.pack(fill="x", side="bottom")

        tk.Label(left, text="Hue Profile", font=("TkDefaultFont", 9, "bold")).pack(anchor="w", side="bottom", pady=(4, 0))
        self.hue_canvas = tk.Canvas(left, width=PREVIEW_MAX, height=50, bg="#121212")
        self.hue_canvas.pack(fill="x", side="bottom")

    def _build_middle_panel(self, parent):
        mid = tk.Frame(parent)
        mid.pack(side="left", fill="y", expand=False, padx=8)

        def num_entry(label, from_, to, init):
            row = tk.Frame(mid)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, width=10, anchor="w").pack(side="left")
            var = NumVar(mid, init, from_, to, self._schedule_update)
            var.make_entry(row, width=8).pack(side="left")
            return var

        # Basic Adjustments
        tk.Label(mid, text="Image Adjustments", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.brightness = num_entry("Brightness", 0.1, 3.0, 1.0)
        self.contrast   = num_entry("Contrast",   0.1, 3.0, 1.0)
        self.saturation = num_entry("Saturation", 0.0, 3.0, 1.0)

        # Filters
        tk.Label(mid, text="Filters", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(16, 4))
        sf = tk.Frame(mid)
        sf.pack(fill="x", pady=1)
        self.sharpen_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sf, text="Enable Sharpening (Unsharp Mask)", variable=self.sharpen_var, 
                       command=self._schedule_update).pack(side="left", anchor="w")

    def _build_right_panel(self, parent):
        right = tk.Frame(parent)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        # Color Curve Section
        tk.Label(right, text="Color Curve (LUTs)", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 4))
        ch_row = tk.Frame(right)
        ch_row.pack(fill="x", pady=2)
        tk.Label(ch_row, text="Channel:").pack(side="left", padx=(0, 4))
        
        self.channel_selector = ttk.Combobox(ch_row, values=["Value", "Red", "Green", "Blue"], width=10, state="readonly")
        self.channel_selector.set("Value")
        self.channel_selector.pack(side="left")
        self.channel_selector.bind("<<ComboboxSelected>>", lambda e: self.curve_widget.set_channel(self.channel_selector.get()))
        
        self.curve_widget = CurveWidget(right, width=256, height=200, callback=self._schedule_update)
        self.curve_widget.pack(fill="x", pady=4)

        # Full Image Preview Section
        tk.Label(right, text="Full Image Preview", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(16, 4))
        self.canvas = tk.Canvas(right, width=TILE_PREVIEW_SIZE, height=TILE_PREVIEW_SIZE, bg="#121212")
        self.canvas.pack(fill="both", expand=True)

    def _build_tile_preview(self, parent):
        tk.Label(parent, text="Tile Inspector", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.preview = tk.Canvas(parent, width=PREVIEW_MAX, height=PREVIEW_MAX, bg="black")
        self.preview.pack(side="top", anchor='nw')
        self.preview.pack_propagate(False)

        btn_cfg = dict(text="", width=2, relief="flat", bd=0,
                       bg="#555555", fg="#555555", activebackground="#888888",
                       font=("TkDefaultFont", 7))
        tk.Button(self.preview, text="^", cnf=btn_cfg, width=1, height=1, bg="gray",
                  command=lambda: self._move_preview_tile(0, -self.preview_move_step)).pack(side="top")
        tk.Button(self.preview, text="~", cnf=btn_cfg, width=1, height=1, bg="gray",
                  command=lambda: self._move_preview_tile(0, self.preview_move_step)).pack(side="bottom")
        tk.Button(self.preview, text="<", cnf=btn_cfg, width=1, height=1, bg="gray",
                  command=lambda: self._move_preview_tile(-self.preview_move_step, 0)).pack(side="left")
        tk.Button(self.preview, text=">", cnf=btn_cfg, width=1, height=1, bg="gray",
                  command=lambda: self._move_preview_tile(self.preview_move_step, 0)).pack(side="right")

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _open_tiff(self):
        path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")])
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

            PREVIEW_SAMPLE_SCALE = 4
            tile_size = TILE_PREVIEW_SIZE * PREVIEW_SAMPLE_SCALE

            cx, cy = full_w // 2, full_h // 2
            x0 = max(0, cx - tile_size // 2)
            y0 = max(0, cy - tile_size // 2)
            x1 = min(full_w, x0 + tile_size)
            y1 = min(full_h, y0 + tile_size)

            tile = self._zarr_level[y0:y1, x0:x1]
            if tile.ndim == 2:
                tile = np.stack([tile, tile, tile], axis=-1)
            elif tile.shape[2] > 3:
                tile = tile[:, :, :3]

            self.preview_base_rgb = to_uint8(tile)
            self.preview_position = (x0, y0, x1, y1)
            self.preview_move_step = TILE_PREVIEW_SIZE

            self.base_rgb = arr
            self.tiff_path = path
            self.file_label.config(text=f"{os.path.basename(path)}  preview {arr.shape[1]}x{arr.shape[0]}  full {base_w}x{base_h}")
            self._schedule_update()
        except Exception as e:
            messagebox.showerror("Error", f"Could not open TIFF:\n{e}")

    def _save_tiff(self):
        if self.base_rgb is None:
            messagebox.showwarning("No image", "Open a TIFF first.")
            return
        out_path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF files", "*.tif *.tiff")])
        if not out_path:
            return

        brightness = self.brightness.get()
        contrast   = self.contrast.get()
        saturation = self.saturation.get()
        luts       = self.curve_widget.get_all_luts()
        sharpen    = self.sharpen_var.get()

        from concurrent.futures import ThreadPoolExecutor

        def process_block(arr_chunk):
            orig_dtype = arr_chunk.dtype
            orig_ndim = arr_chunk.ndim
            
            if orig_ndim == 2:
                working_arr = np.stack([arr_chunk, arr_chunk, arr_chunk], axis=-1)
            elif arr_chunk.shape[2] > 3:
                working_arr = arr_chunk[:, :, :3]
            else:
                working_arr = arr_chunk
            
            working_arr = to_uint8(working_arr)
            
            adj_pil = apply_adjustments(
                working_arr, 1.0, 1.0, 1.0,  # Default RGB gains
                brightness, contrast, saturation,
                luts=luts, sharpen=sharpen
            )
            adj_arr = np.array(adj_pil)
            
            if np.issubdtype(orig_dtype, np.integer):
                max_val = np.iinfo(orig_dtype).max
                adjusted_rgb = (adj_arr.astype(np.float32) / 255.0 * max_val).astype(orig_dtype)
            else:
                adjusted_rgb = adj_arr.astype(orig_dtype)
            
            if orig_ndim == 2:
                return adjusted_rgb
            elif arr_chunk.shape[2] > 3:
                extra_channels = arr_chunk[:, :, 3:]
                return np.concatenate([adjusted_rgb, extra_channels], axis=-1)
            else:
                return adjusted_rgb

        try:
            messagebox.showinfo("Saving", "WARNING: This could take a couple minutes...")
            with tifffile.TiffFile(self.tiff_path) as tif:
                is_bigtiff = tif.is_bigtiff
                with tifffile.TiffWriter(out_path, bigtiff=is_bigtiff) as tw:
                    
                    def process_and_write_page(page, is_subifd=False):
                        arr = page.asarray()
                        h, w = arr.shape[0], arr.shape[1]
                        final_arr = np.empty_like(arr)
                        block_size = 4096
                        slices = []
                        for y in range(0, h, block_size):
                            for x in range(0, w, block_size):
                                slices.append((slice(y, min(y + block_size, h)), slice(x, min(x + block_size, w))))
                        
                        with ThreadPoolExecutor() as executor:
                            futures = {executor.submit(process_block, arr[slc]): slc for slc in slices}
                            for future in futures:
                                slc = futures[future]
                                final_arr[slc] = future.result()
                        
                        write_kwargs = {}
                        if page.is_tiled:
                            write_kwargs['tile'] = (page.tilewidth, page.tilelength)
                        if page.compression:
                            write_kwargs['compression'] = page.compression
                        if hasattr(page, 'subfiletype'):
                            write_kwargs['subfiletype'] = page.subfiletype
                        if arr.ndim == 2:
                            write_kwargs['photometric'] = 'rgb'
                        elif hasattr(page, 'photometric'):
                            write_kwargs['photometric'] = page.photometric
                        if hasattr(page, 'planarconfig'):
                            write_kwargs['planarconfig'] = page.planarconfig
                        
                        has_subifds = hasattr(page, 'subifds') and page.subifds
                        if not is_subifd and has_subifds:
                            write_kwargs['subifds'] = len(page.subifds)
                        
                        tw.write(final_arr, **write_kwargs)
                        if not is_subifd and has_subifds:
                            for sub_page in page.subifds:
                                process_and_write_page(sub_page, is_subifd=True)

                    for page in tif.pages:
                        process_and_write_page(page, is_subifd=False)

            messagebox.showinfo("Saved", f"Saved successfully to:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save properly:\n{e}")

    # ------------------------------------------------------------------
    # Preview rendering
    # ------------------------------------------------------------------

    def _get_adjusted_pil(self):
        return apply_adjustments(
            self.base_rgb,
            1.0, 1.0, 1.0,  # default rgb gains
            self.brightness.get(), self.contrast.get(), self.saturation.get(),
            luts=self.curve_widget.get_all_luts(),
            sharpen=self.sharpen_var.get()
        )

    def _schedule_update(self):
        if self._update_job:
            self.after_cancel(self._update_job)
        self._update_job = self.after(60, self._update_preview)

    def _update_preview(self):
        if self.base_rgb is None:
            return
        pil = self._get_adjusted_pil()
        bh, bw = self.base_rgb.shape[0], self.base_rgb.shape[1]
        w, h = pil.size
        
        cw = self.canvas.winfo_width() or TILE_PREVIEW_SIZE
        ch = self.canvas.winfo_height() or TILE_PREVIEW_SIZE
        
        scale = min(cw / w, ch / h, 1.0)
        pw, ph = max(1, int(w * scale)), max(1, int(h * scale))
        thumb = pil.resize((pw, ph), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(thumb)
        self.canvas.delete("all")

        img_x_off = max(0, (cw - pw) / 2)
        img_y_off = max(0, (ch - ph) / 2)
        self.canvas.create_image(img_x_off, img_y_off, anchor="nw", image=self._tk_img)

        if hasattr(self, "preview_position") and self._zarr_level is not None:
            zfh, zfw = self._zarr_level.shape[0], self._zarr_level.shape[1]
            ds = min(cw / bw, ch / bh, 1.0)
            sx, sy = bw / zfw, bh / zfh
            tx0, ty0, tx1, ty1 = self.preview_position
            rx0 = tx0 * sx * ds + img_x_off
            ry0 = ty0 * sy * ds + img_y_off
            rx1 = tx1 * sx * ds + img_x_off
            ry1 = ty1 * sy * ds + img_y_off
            self.canvas.create_rectangle(rx0 - 1, ry0 - 1, rx1 + 1, ry1 + 1, outline="black", width=3)
            self.canvas.create_rectangle(rx0, ry0, rx1, ry1, outline="yellow", width=1)

        hcw = self.hist_canvas.winfo_width() or PREVIEW_MAX
        huw = self.hue_canvas.winfo_width() or PREVIEW_MAX
        draw_histogram(self.hist_canvas, thumb, hcw, 80)
        draw_hue_profile(self.hue_canvas, thumb, huw, 50)

        if hasattr(self, "preview_base_rgb") and self.preview_base_rgb is not None:
            tile_pil = apply_adjustments(
                self.preview_base_rgb,
                1.0, 1.0, 1.0, 
                self.brightness.get(), self.contrast.get(), self.saturation.get(),
                luts=self.curve_widget.get_all_luts(),
                sharpen=self.sharpen_var.get()
            )
            pw2 = self.preview.winfo_width()  or PREVIEW_MAX
            ph2 = self.preview.winfo_height() or PREVIEW_MAX
            tile_pil = tile_pil.resize((pw2, ph2), Image.LANCZOS)
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

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset(self):
        for var in (self.brightness, self.contrast, self.saturation):
            var.set(1.0)
        self.sharpen_var.set(False)
        self.curve_widget.reset_all()
        self.channel_selector.set("Value")
        self._schedule_update()