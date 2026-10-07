import sys
import math
import numpy as np
import cv2
sys.path.insert(0, '/home/wj/LIMO_GAZEBO/ws_limo_humble/complete')
import race_node_complete as rn

MAP = '/home/wj/LIMO_GAZEBO/ws_limo_humble/maps/track_newmap.yaml'
WORLD = '/home/wj/LIMO_GAZEBO/ws_limo_humble/src/limo_ros2/limo_car/worlds/roboracer/ifac_roboracer.world'
N = 400
KMAX = math.tan(0.42) / 0.24

free_mask, occ_mask, res, origin = rn.load_occupancy(MAP)
walls = rn.load_world_walls(WORLD)
comb = rn.rasterize_walls(free_mask, res, origin, walls)
ox, oy = float(origin[0]), float(origin[1])
H, W = comb.shape[:2]


def build(src_mask):
    raw = rn.extract_centerline(src_mask, res, origin)
    raw = rn.project_clear(raw, comb, res, origin)
    gaps = np.linalg.norm(np.diff(raw, axis=0, append=raw[0:1]), axis=1)
    raw = raw[gaps > 1e-4]
    c = rn.resample_closed_curve(raw, N, smooth_factor=0.02 * len(raw))
    c = rn.resample_closed_curve(c, N, smooth_factor=0.01 * N)
    return c


def recenter(pts, rounds, sf):
    for _ in range(rounds):
        tangent, normals = rn.compute_normals(pts)
        lb, rb = rn.compute_track_bounds(pts, normals, comb, res, origin)
        pts = pts + normals * (0.5 * (np.asarray(lb) - np.asarray(rb)))[:, None]
        pts = rn.resample_closed_curve(pts, N, smooth_factor=sf)
    return pts


def metrics(pts):
    kappa, ds, heading = rn.path_curvature(pts)
    clr = np.array([rn.wall_clearance(walls, x, y) for x, y in pts])
    return clr, kappa, heading, ds.sum()


B = build(comb)
best = None
for rounds in (1, 2, 3, 5):
    for f in (0.01, 0.03, 0.05, 0.1, 0.2, 0.4):
        try:
            p = recenter(B, rounds, f * N)
        except Exception as e:
            print(f'rounds {rounds} sf {f}: failed {e!r}')
            continue
        clr, kappa, heading, L = metrics(p)
        nk = int((np.abs(kappa) > KMAX).sum())
        print(f'rounds {rounds} sf {f}: min clearance {clr.min():.3f}, max|kappa| {np.abs(kappa).max():.2f}, over-limit pts {nk}, length {L:.2f}')
        key = (nk == 0, -nk, float(clr.min()))
        if best is None or key > best[0]:
            best = (key, rounds, f, p)

key, rounds, f, P = best
clr, kappa, heading, L = metrics(P)
d = np.hypot(P[:, 0] - 0.0, P[:, 1] - 1.0)
j = int(np.argmin(d))
print(f'CHOSEN rounds {rounds} sf {f}: min clearance {clr.min():.3f}, max|kappa| {np.abs(kappa).max():.2f} (limit {KMAX:.2f}), over-limit pts {int((np.abs(kappa) > KMAX).sum())}, length {L:.2f}')
print(f'CHOSEN from spawn (0,1): dist {d[j]:.2f} m, heading {math.degrees(heading[j]):.1f} deg')
low = np.where(clr < 0.2)[0]
for i in low:
    print(f'  narrow idx {i}: ({P[i,0]:.2f},{P[i,1]:.2f}) clearance {clr[i]:.3f}')
np.savetxt('/root/path_C.csv', P, delimiter=',', fmt='%.4f')
print('saved /root/path_C.csv')

S = 4
img = np.zeros((H, W, 3), np.uint8)
fm = free_mask.astype(bool)
cm = comb.astype(bool)
img[fm] = (255, 255, 255)
img[fm & ~cm] = (0, 0, 200)
img = cv2.resize(img, (W * S, H * S), interpolation=cv2.INTER_NEAREST)


def px(x, y):
    return int((x - ox) / res * S), int((H - 1 - (y - oy) / res) * S)


poly = np.array([px(x, y) for x, y in P], np.int32)
cv2.polylines(img, [poly], True, (0, 160, 0), 2)
for i in np.where(np.abs(kappa) > KMAX)[0]:
    cv2.circle(img, px(P[i, 0], P[i, 1]), 6, (255, 0, 255), -1)
for i in low:
    cv2.circle(img, px(P[i, 0], P[i, 1]), 6, (0, 140, 255), -1)
cv2.circle(img, px(0.0, 1.0), 8, (0, 200, 255), -1)
cv2.imwrite('/root/path_D.png', img)
print('saved /root/path_D.png')
