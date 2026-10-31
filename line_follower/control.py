"""ROS-free controller: grayscale image -> (linear, angular)."""
import math
import numpy as np
from line_follower import vision

DEFAULTS = dict(
    line_polarity='dark', dark_frac=0.65, dark_thresh=0.0, min_contrast=5.0,
    group_rows=3, min_w=0.015, max_w=0.45, ref_tol=0.35, jump_tol=0.25, kink_angle=0.45,
    # Safe increased cruise speed (0.14 m/s) with stable corner handling
    v_max=0.14, v_min=0.025, v_corner=0.04, v_blind=0.05, w_max=2.4,
    kp_y=5.5, kd_psi=2.0, y_clip=0.10, smooth=0.5,
    corner_coast_dist=0.08,
    pivot_w=1.2,
    pivot_min_angle=1.40,
    pivot_max_side=1.85,
    fit_xmax=0.50,
)


def clip(x, lo, hi):
    return max(lo, min(hi, x))


class LineController:
    def __init__(self, **params):
        self.p = dict(DEFAULTS)
        if 'line_polarity' in params:
            self.p['line_polarity'] = params['line_polarity']

        self.state = 'wait'
        self.last_y = 0.0
        self.fit = None
        self.fit_t = -1e9
        self.prev_t = None
        self.last_cmd = (0.0, 0.0)
        self.spawn_t0 = None
        self.seen_line = False

        self.turn_dir = 1.0
        self.coast_remaining = 0.0
        self.pivot_rotated = 0.0
        self.pivot_phase = 1
        self.lost_frames = 0

    def _fit(self, pts, t):
        p = self.p
        sel = pts[pts[:, 0] <= p['fit_xmax']]
        if len(sel) < 2:
            sel = pts
        x, y = sel[:, 0], sel[:, 1]

        n_near = min(5, len(pts))
        y0 = float(np.mean(pts[:n_near, 1]))

        if len(sel) >= 3 and (x.max() - x.min()) >= 0.05:
            c1, _ = np.polyfit(x, y, 1)
            psi = math.atan(c1)
        else:
            psi = 0.0

        new = (y0, psi, 0.0)
        if self.fit is not None and t - self.fit_t < 0.5:
            k = p['smooth']
            new = tuple(k * n + (1 - k) * o for n, o in zip(new, self.fit))
        self.fit, self.fit_t = new, t
        return new

    def _track_cmd(self, y0, psi):
        p = self.p
        y0_clamped = clip(y0, -p['y_clip'], p['y_clip'])
        w = p['kd_psi'] * psi + p['kp_y'] * y0_clamped

        err = abs(psi) / 0.30 + abs(y0_clamped) / 0.04
        v = p['v_max'] / (1.0 + 1.5 * err)
        v = max(p['v_min'], v)

        return v, clip(w, -p['w_max'], p['w_max'])

    def _out(self, v, w):
        self.last_cmd = (v, w)
        return v, w

    def update(self, gray, t, m):
        p = self.p
        if self.spawn_t0 is None:
            self.spawn_t0 = t

        dt = 0.05 if self.prev_t is None else clip(t - self.prev_t, 1e-3, 0.5)
        self.prev_t = t

        if (t - self.spawn_t0) < 1.5:
            return self._out(0.0, 0.0)

        tr = vision.extract(gray, m, p, self.last_y)
        pts = tr['pts']
        have = len(pts) >= 2

        if self.state == 'coast':
            self.coast_remaining -= p['v_blind'] * dt
            if self.coast_remaining <= 0.0:
                self.state = 'pivot'
                self.pivot_rotated = 0.0
                self.pivot_phase = 1
                return self._out(0.0, 0.0)
            return self._out(p['v_blind'], 0.0)

        if self.state == 'pivot':
            if self.pivot_phase == 1:
                w_cmd = self.turn_dir * p['pivot_w']
                self.pivot_rotated += abs(w_cmd) * dt

                if self.pivot_rotated >= p['pivot_min_angle'] and have:
                    near_y = float(np.mean(pts[:min(4, len(pts)), 1]))
                    if abs(near_y) <= 0.055:
                        self.state = 'track'
                        self.last_y = near_y
                        self.fit = None
                        self.lost_frames = 0
                        return self._out(p['v_min'], 0.0)
                    return self._out(0.0, self.turn_dir * 0.6)

                if self.pivot_rotated >= p['pivot_max_side']:
                    self.pivot_phase = 2
                    self.pivot_rotated = 0.0
                    return self._out(0.0, 0.0)

                return self._out(0.0, w_cmd)
            else:
                w_cmd = -self.turn_dir * p['pivot_w']
                self.pivot_rotated += abs(w_cmd) * dt

                if self.pivot_rotated >= 2.55 and have:
                    near_y = float(np.mean(pts[:min(4, len(pts)), 1]))
                    if abs(near_y) <= 0.055:
                        self.state = 'track'
                        self.last_y = near_y
                        self.fit = None
                        self.lost_frames = 0
                        return self._out(p['v_min'], 0.0)
                    return self._out(0.0, -self.turn_dir * 0.6)

                if self.pivot_rotated > 3.60:
                    self.state = 'track'
                    self.last_y = 0.0
                    self.lost_frames = 0
                    return self._out(0.0, 0.0)

                return self._out(0.0, w_cmd)

        if have:
            self.lost_frames = 0
            self.seen_line = True
            self.state = 'track'
            self.last_y = float(pts[0, 1])
            y0, psi, _ = self._fit(pts, t)

            tip_diff = float(pts[-1, 1] - pts[0, 1])
            if abs(tip_diff) > 0.015:
                self.turn_dir = 1.0 if tip_diff >= 0.0 else -1.0
            elif abs(psi) > 0.04:
                self.turn_dir = 1.0 if psi >= 0.0 else -1.0

            v, w = self._track_cmd(y0, psi)
            return self._out(v, w)

        if self.seen_line:
            self.lost_frames += 1
            if self.lost_frames < 4:
                return self._out(p['v_min'], 0.0)

            self.state = 'coast'
            self.coast_remaining = p['corner_coast_dist']
            return self._out(p['v_blind'], 0.0)

        return self._out(0.0, 0.0)
