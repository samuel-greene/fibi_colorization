import numpy as np
from PIL import Image, ImageEnhance


def to_uint8(arr):
    a = arr.astype(np.float32)
    mn, mx = a.min(), a.max()
    if mx > mn:
        a = (a - mn) / (mx - mn) * 255
    else:
        a = np.zeros_like(a)
    return a.astype(np.uint8)


def apply_adjustments(base_rgb, r_gain, g_gain, b_gain, brightness, contrast, saturation,
                      hue_shift=0.0, hue_min=0.0, hue_max=360.0):
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

    # Hue range masking: desaturate pixels whose hue falls outside [hue_min, hue_max].
    # Default (0, 360) means no masking — all hues pass.
    lo, hi = hue_min % 360, hue_max % 360
    if not (lo == 0.0 and hi == 0.0) and (lo, hi) != (0.0, 360.0):
        hsv8 = np.array(pil.convert("HSV"), dtype=np.float32)
        hue_deg = hsv8[:, :, 0] / 255.0 * 360.0
        if lo <= hi:
            in_range = (hue_deg >= lo) & (hue_deg <= hi)
        else:
            # wraps around 0°/360° (e.g. 330→30)
            in_range = (hue_deg >= lo) | (hue_deg <= hi)
        gray = np.array(pil.convert("L"), dtype=np.uint8)
        gray_rgb = np.stack([gray, gray, gray], axis=-1)
        result = np.array(pil, dtype=np.uint8).copy()
        result[~in_range] = gray_rgb[~in_range]
        pil = Image.fromarray(result, "RGB")

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
    hist = np.log1p(hist)
    if hist.max() > 0:
        hist = hist / hist.max()
    pts = [0, h]
    for x, v in enumerate(hist):
        pts.extend([x, int(h - v * h * 0.9)])
    pts.extend([w, h])
    canvas.create_polygon(pts, fill="white", outline="", stipple="gray75")