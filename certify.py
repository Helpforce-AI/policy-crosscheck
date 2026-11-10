"""Check a policy against the Helpforce Validated protocol HV-LOCO-1 v1.0.

  python certify.py policies/helpforce_go2_v1.onnx                  # runs 80 trials, about 3 minutes on a CPU
  python certify.py policies/your_policy.onnx --out report.json
  python certify.py policies/your_policy.onnx --results run.json    # score an existing 10-seed run instead

The protocol: https://www.helpforce.ai/validated/protocol
  4 tests x 10 seeds x 2 engines (MuJoCo and Genesis) = 80 trials of 6 s each.
  C1  No falls in any trial, in either engine.
  C2  Every trial covers at least 70% of the commanded distance.
  C3  In MuJoCo, median body clearance is at least 50 mm in every test, and no trial dips below 20 mm.

This script tells you whether a policy meets the protocol. A Helpforce Validated certificate is
issued only by Helpforce AI after it reruns the check itself; see the protocol page.
This is simulation evidence. It is not a safety certification and does not replace hardware tests.
"""
import os, sys, json, hashlib, argparse, subprocess, platform, datetime
import numpy as np

PROTOCOL, VERSION = 'HV-LOCO-1', '1.0'
SEEDS, MIN_PROGRESS, MIN_MEDIAN_MM, MIN_ANY_MM = 10, 0.70, 50.0, 20.0
HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def versions():
    import mujoco, onnxruntime
    v = dict(mujoco=mujoco.__version__, onnxruntime=onnxruntime.__version__, python=platform.python_version())
    try:
        import genesis; v['genesis'] = genesis.__version__
    except Exception: v['genesis'] = None
    try: v['runner_commit'] = subprocess.check_output(['git', '-C', HERE, 'rev-parse', '--short', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: v['runner_commit'] = None
    return v


def score(results):
    from crosscheck import TESTS, T
    speed = {name: vx for name, vx, _, _ in TESTS}
    tests, fails = [], []
    for engine in ('mujoco', 'genesis'):
        if engine not in results: fails.append(f'{engine}: not run'); continue
        for name, rs in results[engine].items():
            if len(rs) < SEEDS: fails.append(f'{engine} / {name}: {len(rs)} seeds, protocol needs {SEEDS}')
            falls = sum(r['fell'] is not None for r in rs)
            progress = max(0.0, min(r['dist'] / (speed[name] * T) for r in rs))
            row = dict(engine=engine, test=name, trials=len(rs), falls=falls, min_progress=round(progress, 3))
            if falls: fails.append(f'C1 {engine} / {name}: {falls} of {len(rs)} fell')
            if progress < MIN_PROGRESS: fails.append(f'C2 {engine} / {name}: a trial covered {progress:.0%} of the distance')
            if engine == 'mujoco':
                ok = [r for r in rs if r['fell'] is None and 'clear_median_mm' in r]
                if ok:
                    med = float(np.median([r['clear_median_mm'] for r in ok])); low = min(r['clear_min_mm'] for r in ok)
                    row.update(clearance_median_mm=round(med, 1), clearance_min_mm=round(low, 1))
                    if med < MIN_MEDIAN_MM: fails.append(f'C3 {name}: median clearance {med:.1f} mm, needs {MIN_MEDIAN_MM:.0f}')
                    if low < MIN_ANY_MM: fails.append(f'C3 {name}: clearance dipped to {low:.1f} mm, needs {MIN_ANY_MM:.0f}')
                else: fails.append(f'C3 {name}: no trial walked the full run, clearance not measurable')
            tests.append(row)
    return tests, fails


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('policy', help='ONNX policy exported from Isaac Lab (input "obs", 48 values; output 12 joint targets)')
    ap.add_argument('--results', help='score an existing crosscheck.py --engine both --seeds 10 JSON instead of running')
    ap.add_argument('--out', help='write the report to this JSON file')
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    if a.results:
        results = json.load(open(a.results))['results']
    else:
        import onnxruntime as ort
        from crosscheck import run_mujoco, run_genesis
        pol = ort.InferenceSession(a.policy)
        results = dict(mujoco=run_mujoco(pol, SEEDS), genesis=run_genesis(pol, SEEDS))
    tests, fails = score(results)
    report = dict(protocol=PROTOCOL, protocol_version=VERSION, checked=datetime.date.today().isoformat(),
                  policy_file=os.path.basename(a.policy), policy_sha256=sha256(a.policy),
                  verdict='meets protocol' if not fails else 'does not meet protocol',
                  failures=fails, tests=tests, environment=versions(), results=results)
    print(f'\n{PROTOCOL} v{VERSION}   {report["policy_file"]}')
    print(f'SHA-256  {report["policy_sha256"]}\n')
    for t in tests:
        clr = f"   clearance median {t['clearance_median_mm']} mm, min {t['clearance_min_mm']} mm" if 'clearance_median_mm' in t else ''
        print(f"  {t['engine']:8s} {t['test']:26s} falls {t['falls']}/{t['trials']}   min progress {t['min_progress']:.0%}{clr}")
    print(f'\nVerdict: {report["verdict"].upper()}')
    for f in fails[:12]: print('  - ' + f)
    if len(fails) > 12: print(f'  ... and {len(fails) - 12} more')
    if a.out: json.dump(report, open(a.out, 'w'), indent=1); print(f'\nReport written to {a.out}')
