"""Run an Isaac Lab Go2 policy, unchanged, in MuJoCo and Genesis and compare the verdicts.

  python crosscheck.py policies/helpforce_go2_v1.onnx                 # MuJoCo, 3 seeds
  python crosscheck.py policies/standard_reward.onnx --engine genesis
  python crosscheck.py policies/your_policy.onnx --engine both --seeds 10

A trial fails if the body touches the ground or tilts past 60 degrees. Each trial is a 6 s walk.
In MuJoCo the script also measures the gap between the underside of the body and the floor
and reports it for the trials that walked the full 6 s.
"""
import os, sys, json, argparse
import numpy as np, onnxruntime as ort
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'crosscheck'))
import mujoco
import mujoco_engine as M

TESTS = [  # name, forward speed (m/s), ground friction, extra mass on the body (kg)
    ('walk 0.5 m/s', 0.5, 1.0, 0.0),
    ('walk 1.0 m/s', 1.0, 1.0, 0.0),
    ('6 kg payload at 0.8 m/s', 0.8, 1.0, 6.0),
    ('friction 0.25 at 0.8 m/s', 0.8, 0.25, 0.0),
]
T, HZ = 6.0, 50

def trial(sim, pol, vx, seed, clearance=None):
    sim.reset(seed); cmd = np.array([vx, 0, 0], np.float32)
    x0 = base_pos(sim)[0]; zs, cs, fell = [], [], None
    for k in range(int(T * HZ)):
        sim.step(pol.run(None, {'obs': sim.obs(cmd)[None]})[0][0])
        zs.append(base_pos(sim)[2])
        if clearance: cs.append(clearance(sim))
        if sim.fallen(): fell = round((k + 1) / HZ, 2); break
    r = dict(seed=seed, fell=fell, dist=round(float(base_pos(sim)[0] - x0), 2), height=round(float(np.mean(zs[-50:])), 3))
    if cs: r.update(clear_min_mm=round(min(cs) * 1000, 1), clear_median_mm=round(float(np.median(cs)) * 1000, 1))
    return r

def base_pos(sim):
    return sim.d.xpos[sim.base] if hasattr(sim, 'd') else sim.base()

def run_mujoco(pol, seeds):
    out = {}
    for name, vx, mu, kg in TESTS:
        m = M.build_model(friction=mu, add_mass=kg); sim = M.Sim(m)
        body = [g for g in range(m.ngeom) if m.geom_bodyid[g] == sim.base and (m.geom_contype[g] | m.geom_conaffinity[g])]
        clr = lambda s: min(mujoco.mj_geomDistance(m, s.d, g, s.floor, 0.5, None) for g in body)
        out[name] = [trial(sim, pol, vx, k, clr) for k in range(seeds)]
    return out

def run_genesis(pol, seeds):
    import genesis as gs
    import genesis_engine as G
    gs.init(backend=gs.cpu, logging_level='warning')
    out = {}
    for name, vx, mu, kg in TESTS:
        sim = G.GSim(mu, kg); out[name] = [trial(sim, pol, vx, k) for k in range(seeds)]
    return out

def table(engine, res):
    print(f'\n{engine}')
    for name, rs in res.items():
        falls = sum(r['fell'] is not None for r in rs)
        extra = ''
        ok = [r for r in rs if r['fell'] is None and 'clear_median_mm' in r]
        if ok: extra = f"   clearance (trials that walked) median {np.median([r['clear_median_mm'] for r in ok]):.1f} mm, min {min(r['clear_min_mm'] for r in ok):.1f} mm"
        print(f'  {name:26s} falls {falls}/{len(rs)}   height {np.mean([r["height"] for r in rs]):.3f} m{extra}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('policy', help='ONNX policy exported from Isaac Lab (input "obs", 48 values; output 12 joint targets)')
    ap.add_argument('--engine', choices=['mujoco', 'genesis', 'both'], default='mujoco')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--out', help='write results to this JSON file')
    a = ap.parse_args()
    pol = ort.InferenceSession(a.policy); res = {}
    if a.engine in ('mujoco', 'both'): res['mujoco'] = run_mujoco(pol, a.seeds); table('MuJoCo', res['mujoco'])
    if a.engine in ('genesis', 'both'): res['genesis'] = run_genesis(pol, a.seeds); table('Genesis', res['genesis'])
    if a.out: json.dump(dict(policy=os.path.basename(a.policy), seeds=a.seeds, results=res), open(a.out, 'w'), indent=1)
