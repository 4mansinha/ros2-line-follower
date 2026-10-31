"""Pure-numpy perception: image -> centre-line points on the ground (metres)."""
import math
import numpy as np


def to_gray(data, height, width, step, encoding, stride=1):
    enc = encoding.lower()
    ch = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1, '8uc1': 1}.get(enc)
    if ch is None:
        return None
    buf = np.frombuffer(bytes(data), dtype=np.uint8)
    step = step if step else width * ch
    if buf.size < height * step:
        return None
    img = buf[:height * step].reshape(height, step)[:, :width * ch].reshape(height, width, ch)
    img = img[::stride, ::stride].astype(np.float32)
    if ch == 1:
        return img[:, :, 0]
    if enc.startswith('rgb'):
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    else:
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def otsu(gray):
    g = np.clip(gray, 0, 255).astype(np.uint8)
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return None, 0.0
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    denom = omega * (1.0 - omega)
    ok = denom > 1e-12
    if not ok.any():
        return None, 0.0
    sig = np.zeros(256)
    sig[ok] = (mu[-1] * omega[ok] - mu[ok]) ** 2 / denom[ok]
    t = int(np.argmax(sig))
    lo, hi = g[g <= t], g[g > t]
    if lo.size == 0 or hi.size == 0:
        return None, 0.0
    return t, float(hi.mean() - lo.mean())


def find_runs(row_mask, min_px=3):
    padded = np.concatenate(([False], row_mask, [False]))
    starts = np.where(padded[1:] & ~padded[:-1])[0]
    ends = np.where(~padded[1:] & padded[:-1])[0] - 1
    return [(int(a), int(b)) for a, b in zip(starts, ends) if (b - a + 1) >= min_px]


class CamModel:
    def __init__(self, width, height, hfov, cam_h, cam_x, pitch, r_max=0.8):
        f = (width / 2.0) / math.tan(hfov / 2.0)
        self.width, self.height, self.cx = width, height, width / 2.0
        b = (np.arange(height) + 0.5 - height / 2.0) / f
        dz = -math.sin(pitch) - b * math.cos(pitch)
        dx = math.cos(pitch) - b * math.sin(pitch)
        ok = dz < -1e-6
        with np.errstate(divide='ignore', invalid='ignore'):
            t = -cam_h / dz
        self.bx = np.where(ok, cam_x + t * dx, np.inf)
        self.s = np.where(ok, t / f, 0.0)
        self.valid = ok & (self.bx <= r_max)
        rows = np.where(self.valid)[0]
        self.r0, self.r1 = int(rows.min()), int(rows.max()) + 1
        self.x_near = float(self.bx[self.r1 - 1])
        bx = np.where(self.valid, self.bx, np.nan)
        dxr = np.nan_to_num(np.abs(np.gradient(bx)))
        self.area = self.s * dxr
        self.cols = np.arange(width) - self.cx


def extract(gray, m, p, ref_y):
    out = dict(pts=np.zeros((0, 3)), fit_pts=np.zeros((0, 3)), x_end=0.0, ended='none')
    roi = gray[m.r0:m.r1]
    thr, contrast = otsu(roi)
    if thr is None:
        return out

    mask = (roi <= thr) if p['line_polarity'] != 'light' else (roi > thr)

    G = int(p['group_rows'])
    nrow = mask.shape[0]
    pts = []
    y_prev = ref_y
    gap = 0

    r = nrow
    while r - G >= 0:
        g0 = r - G
        rows = slice(m.r0 + g0, m.r0 + r)
        r = g0

        x = float(m.bx[rows].mean())
        s = float(m.s[rows].mean())

        if not pts and x > m.x_near + 0.18:
            break

        row_mask = mask[g0:g0 + G].mean(axis=0) >= 0.4
        runs = find_runs(row_mask, min_px=3)

        cands = []
        for a, b in runs:
            yl = -(a - m.cx) * s
            yr = -(b - m.cx) * s
            yc = 0.5 * (yl + yr)
            wm = yl - yr
            if p['min_w'] <= wm <= p['max_w']:
                if not pts:
                    cands.append((abs(yc - ref_y), yc, wm))
                else:
                    if (yr - 0.03 <= y_prev <= yl + 0.03) or abs(yc - y_prev) <= 0.09:
                        cands.append((abs(yc - y_prev), yc, wm))

        if not cands:
            if pts:
                gap += 1
                if gap > 2:
                    break
            continue

        gap = 0
        cands.sort(key=lambda item: item[0])
        _, yc, wm = cands[0]
        pts.append((x, yc, wm))
        y_prev = yc

    if len(pts) < 2:
        return out

    P = np.array(pts)
    out['pts'] = P
    out['fit_pts'] = P
    out['x_end'] = float(P[-1, 0])
    return out
