import sys
import math
import numpy as np
sys.path.insert(0, '/home/wj/LIMO_GAZEBO/ws_limo_humble/complete')
import race_node_complete as rn

MAP = '/home/wj/LIMO_GAZEBO/ws_limo_humble/maps/track_newmap.yaml'
WORLD = '/home/wj/LIMO_GAZEBO/ws_limo_humble/src/limo_ros2/limo_car/worlds/roboracer/ifac_roboracer.world'
N = 400

free_mask, occ_mask, res, origin = rn.load_occupancy(MAP)
walls = rn.load_world_walls(WORLD)
comb = rn.rasterize_walls(free_mask, res, origin, walls)


def build(src_mask):
    raw = rn.extract_centerline(src_mask, res, origin)
    raw = rn.project_clear(raw, comb, res, origin)
    gaps = np.linalg.norm(np.diff(raw, axis=0, append=raw[0:1]), axis=1)
    raw = raw[gaps > 1e-4]
    c = rn.resample_closed_curve(raw, N, smooth_factor=0.02 * len(raw))
    c = rn.resample_closed_curve(c, N, smooth_factor=0.01 * N)
    return c


def report(name, pts):
    kappa, ds, heading = rn.path_curvature(pts)
    clr = np.array([rn.wall_clearance(walls, x, y) for x, y in pts])
    print(f'=== {name}: length {ds.sum():.2f} m, max|kappa| {np.abs(kappa).max():.3f}, min clearance {clr.min():.3f} m')
    bad = np.where(clr < 0.15)[0]
    print(f'points with clearance < 0.15 m: {len(bad)} / {len(pts)}')
    if len(bad):
        segs = []
        start = prev = bad[0]
        for i in bad[1:]:
            if i != prev + 1:
                segs.append((start, prev))
                start = i
            prev = i
        segs.append((start, prev))
        for a, b in segs:
            seg = clr[a:b + 1]
            j = a + int(np.argmin(seg))
            print(f'  idx {a}-{b}: worst at ({pts[j,0]:.2f},{pts[j,1]:.2f}) clearance {clr[j]:.3f}')
    for sx, sy, syaw, label in [(0.0, 1.0, 0.0, 'default spawn (0,1,yaw0)'), (-0.2, 0.8, 0.3491, 'complete spawn (-0.2,0.8,yaw20deg)')]:
        d = np.hypot(pts[:, 0] - sx, pts[:, 1] - sy)
        k = int(np.argmin(d))
        diff = math.degrees(math.atan2(math.sin(heading[k] - syaw), math.cos(heading[k] - syaw)))
        print(f'  {label}: nearest idx {k} dist {d[k]:.2f} m, path heading {math.degrees(heading[k]):.1f} deg, diff from car yaw {diff:.1f} deg, clearance there {clr[k]:.3f}')


report('A: current (centerline from map only)', build(free_mask))
try:
    report('B: centerline from map + world walls', build(comb))
except Exception as e:
    print('B failed:', repr(e))
