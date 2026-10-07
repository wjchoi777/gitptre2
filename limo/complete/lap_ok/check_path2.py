import sys
import math
import numpy as np
import cv2
sys.path.insert(0, '/home/wj/LIMO_GAZEBO/ws_limo_humble/complete')
import race_node_complete as rn

MAP = '/home/wj/LIMO_GAZEBO/ws_limo_humble/maps/track_newmap.yaml'
WORLD = '/home/wj/LIMO_GAZEBO/ws_limo_humble/src/limo_ros2/limo_car/worlds/roboracer/ifac_roboracer.world'
N = 400

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


def recenter(pts, rounds):
    for _ in range(rounds):
        tangent, normals = rn.compute_normals(pts)
        lb, rb = rn.compute_track_bounds(pts, normals, comb, res, origin)
        pts = pts + normals * (0.5 * (np.asarray(lb) - np.asarray(rb)))[:, None]
        pts = rn.resample_closed_curve(pts, N, smooth_factor=0.01 * N)
    return pts


def report(name, pts):
    kappa, ds, heading = rn.path_curvature(pts)
    clr = np.array([rn.wall_clearance(walls, x, y) for x, y in pts])
    k = int(np.argmax(np.abs(kappa)))
    print(f'=== {name}: length {ds.sum():.2f} m, max|kappa| {abs(kappa[k]):.3f} at idx {k} ({pts[k,0]:.2f},{pts[k,1]:.2f}), min clearance {clr.min():.3f} m at idx {int(np.argmin(clr))}')
    print(f'    points < 0.15 m: {int((clr < 0.15).sum())}, < 0.25 m: {int((clr < 0.25).sum())}, |kappa| > 1.5: {int((np.abs(kappa) > 1.5).sum())}')
    d = np.hypot(pts[:, 0] - 0.0, pts[:, 1] - 1.0)
    j = int(np.argmin(d))
    print(f'    from spawn (0,1): dist {d[j]:.2f} m, heading {math.degrees(heading[j]):.1f} deg')


def to_px(pts, flip):
    cols = (pts[:, 0] - ox) / res
    rows = (pts[:, 1] - oy) / res
    if flip:
        rows = H - 1 - rows
    return cols, rows


A = build(free_mask)
B = build(comb)
C = recenter(B, 3)
report('A map only', A)
report('B map+world', B)
report('C B recentered x3', C)

best_flip, best_score = False, -1.0
for flip in (False, True):
    cols, rows = to_px(B, flip)
    ci = np.clip(cols.astype(int), 0, W - 1)
    ri = np.clip(rows.astype(int), 0, H - 1)
    score = float(comb[ri, ci].astype(bool).mean())
    if score > best_score:
        best_flip, best_score = flip, score
print(f'image row flip={best_flip}, path-on-free ratio {best_score:.2f}')

S = 4
img = np.zeros((H, W, 3), np.uint8)
fm = free_mask.astype(bool)
cm = comb.astype(bool)
img[fm] = (255, 255, 255)
img[fm & ~cm] = (0, 0, 200)
if best_flip:
    pass
img = cv2.resize(img, (W * S, H * S), interpolation=cv2.INTER_NEAREST)
for pts, color in ((A, (0, 0, 255)), (B, (255, 0, 0)), (C, (0, 160, 0))):
    cols, rows = to_px(pts, best_flip)
    poly = np.stack([cols * S, rows * S], axis=1).astype(np.int32)
    cv2.polylines(img, [poly], True, color, 2)
cols, rows = to_px(np.array([[0.0, 1.0]]), best_flip)
cv2.circle(img, (int(cols[0] * S), int(rows[0] * S)), 8, (0, 200, 255), -1)
cv2.imwrite('/root/path_check.png', img)
print('saved /root/path_check.png')
