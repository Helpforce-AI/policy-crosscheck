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

# name, forward speed (m/s), ground friction, extra mass on the body (kg), extras
# extras: push = sideways velocity kick in m/s at 2 s and 4 s, delay = policy steps of action latency (1 = 20 ms),
#         noise = Isaac Lab's default observation noise
TESTS = [
    ('walk 0.5 m/s', 0.5, 1.0, 0.0, {}),
    ('walk 1.0 m/s', 1.0, 1.0, 0.0, {}),
    ('6 kg payload at 0.8 m/s', 0.8, 1.0, 6.0, {}),
    ('friction 0.25 at 0.8 m/s', 0.8, 0.25, 0.0, {}),
]
STRESS = [  # added in protocol HV-LOCO-1 v1.1
    ('push 1.0 m/s at 0.5 m/s', 0.5, 1.0, 0.0, {'push': 1.0}),
    ('10 kg payload at 0.5 m/s', 0.5, 1.0, 10.0, {}),
    ('20 ms latency at 0.8 m/s', 0.8, 1.0, 0.0, {'delay': 1}),
    ('sensor noise at 0.8 m/s', 0.8, 1.0, 0.0, {'noise': True}),
]
# Isaac Lab velocity task default observation noise (uniform, +/-): lin vel, ang vel, gravity, cmd, joint pos, joint vel, last action
NOISE = np.concatenate([np.full(3, 0.1), np.full(3, 0.2), np.full(3, 0.05), np.zeros(3), np.full(12, 0.01), np.full(12, 1.5), np.zeros(12)]).astype(np.float32)
T, HZ = 6.0, 50

def trial(sim, pol, vx, seed, clearance=None, extra=None):
    extra = extra or {}
    sim.reset(seed); cmd = np.array([vx, 0, 0], np.float32)
    rng = np.random.default_rng(5000 + seed)
    kicks = {int(2.0 * HZ): rng.uniform(0, 2 * np.pi), int(4.0 * HZ): rng.uniform(0, 2 * np.pi)} if extra.get('push') else {}
    queue = [np.zeros(12, np.float32)] * extra.get('delay', 0)
    x0 = base_pos(sim)[0]; zs, cs, fell = [], [], None
    for k in range(int(T * HZ)):
        if k in kicks: kick(sim, extra['push'] * np.cos(kicks[k]), extra['push'] * np.sin(kicks[k]))
        o = sim.obs(cmd)
        if extra.get('noise'): o = o + rng.uniform(-1, 1, o.shape).astype(np.float32) * NOISE
        a = pol.run(None, {'obs': o[None]})[0][0]
        if queue: queue.append(a); a = queue.pop(0)
        sim.step(a)
        zs.append(base_pos(sim)[2])
        if clearance: cs.append(clearance(sim))
        if sim.fallen(): fell = round((k + 1) / HZ, 2); break
    r = dict(seed=seed, fell=fell, dist=round(float(base_pos(sim)[0] - x0), 2), height=round(float(np.mean(zs[-50:])), 3))
    if cs: r.update(clear_min_mm=round(min(cs) * 1000, 1), clear_median_mm=round(float(np.median(cs)) * 1000, 1))
    return r

def kick(sim, dvx, dvy):
    if hasattr(sim, 'd'): sim.d.qvel[0] += dvx; sim.d.qvel[1] += dvy
    else: sim.kick(dvx, dvy)

def base_pos(sim):
    return sim.d.xpos[sim.base] if hasattr(sim, 'd') else sim.base()

def run_mujoco(pol, seeds, tests=None):
    out = {}
    for name, vx, mu, kg, ex in (tests or TESTS):
        m = M.build_model(friction=mu, add_mass=kg); sim = M.Sim(m)
        body = [g for g in range(m.ngeom) if m.geom_bodyid[g] == sim.base and (m.geom_contype[g] | m.geom_conaffinity[g])]
        clr = lambda s: min(mujoco.mj_geomDistance(m, s.d, g, s.floor, 0.5, None) for g in body)
        out[name] = [trial(sim, pol, vx, k, clr, ex) for k in range(seeds)]
    return out

def run_genesis(pol, seeds, tests=None):
    import genesis as gs
    import genesis_engine as G
    try: gs.init(backend=gs.cpu, logging_level='warning')
    except Exception: pass  # already initialised
    out = {}
    for name, vx, mu, kg, ex in (tests or TESTS):
        sim = G.GSim(mu, kg); out[name] = [trial(sim, pol, vx, k, None, ex) for k in range(seeds)]
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
    ap.add_argument('--stress', action='store_true', help='also run the four stress tests added in protocol v1.1')
    ap.add_argument('--out', help='write results to this JSON file')
    a = ap.parse_args()
    pol = ort.InferenceSession(a.policy); res = {}; tests = TESTS + (STRESS if a.stress else [])
    if a.engine in ('mujoco', 'both'): res['mujoco'] = run_mujoco(pol, a.seeds, tests); table('MuJoCo', res['mujoco'])
    if a.engine in ('genesis', 'both'): res['genesis'] = run_genesis(pol, a.seeds, tests); table('Genesis', res['genesis'])
    if a.out: json.dump(dict(policy=os.path.basename(a.policy), seeds=a.seeds, results=res), open(a.out, 'w'), indent=1)
