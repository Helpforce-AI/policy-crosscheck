"""Check a policy against the Helpforce Validated protocol HV-LOCO-1 v1.0.

  python certify.py policies/helpforce_go2_v1.onnx                  # runs 80 trials, about 3 minutes on a CPU
  python certify.py policies/your_policy.onnx --out report.json
  python certify.py policies/your_policy.onnx --results run.json    # score an existing 10-seed run instead

The protocol: https://www.helpforce.ai/validated/protocol
  Core: 4 tests x 10 seeds x 2 engines (MuJoCo and Genesis) = 80 trials of 6 s each. All must pass.
    C1  No falls in any core trial, in either engine.
    C2  Every core trial covers at least 70% of the commanded distance.
    C3  In MuJoCo, median body clearance is at least 50 mm in every core test, and no trial dips below 20 mm.
  Stress (added in v1.1): pushes, 10 kg payload, 20 ms latency, sensor noise. Another 80 trials.
    S   A stress test passes when at most 1 of 10 trials falls in each engine. Stress results are graded
        and published on the certificate; they do not block it.

This script tells you whether a policy meets the protocol. A Helpforce Validated certificate is
issued only by Helpforce AI after it reruns the check itself; see the protocol page.
This is simulation evidence. It is not a safety certification and does not replace hardware tests.
"""
import os, sys, json, hashlib, argparse, subprocess, platform, datetime
import numpy as np

PROTOCOL, VERSION = 'HV-LOCO-1', '1.1'
STRESS_MAX_FALLS = 1
SEEDS, MIN_PROGRESS, MIN_MEDIAN_MM, MIN_ANY_MM = 10, 0.70, 50.0, 20.0
HERE = os.path.dirname(os.path.abspath(__file__))


def STRESS_NAMES(stress): return sorted({x['test'] for x in stress})


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
    from crosscheck import TESTS, STRESS, T
    speed = {t[0]: t[1] for t in TESTS + STRESS}
    core = {t[0] for t in TESTS}; stress_names = [t[0] for t in STRESS]
    tests, fails, stress = [], [], []
    for engine in ('mujoco', 'genesis'):
        if engine not in results: fails.append(f'{engine}: not run'); continue
        for name in stress_names:
            rs = results[engine].get(name)
            if rs is None: stress.append(dict(engine=engine, test=name, trials=0, falls=None, passed=None)); continue
            f = sum(r['fell'] is not None for r in rs)
            stress.append(dict(engine=engine, test=name, trials=len(rs), falls=f, passed=f <= STRESS_MAX_FALLS))
        for name, rs in results[engine].items():
            if name not in core: continue
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
    return tests, fails, stress


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
        from crosscheck import TESTS, STRESS
        results = dict(mujoco=run_mujoco(pol, SEEDS, TESTS + STRESS), genesis=run_genesis(pol, SEEDS, TESTS + STRESS))
    tests, fails, stress = score(results)
    report = dict(protocol=PROTOCOL, protocol_version=VERSION, checked=datetime.date.today().isoformat(),
                  policy_file=os.path.basename(a.policy), policy_sha256=sha256(a.policy),
                  verdict='meets core protocol' if not fails else 'does not meet core protocol',
                  failures=fails, tests=tests, stress=stress,
                  stress_passed=sum(1 for n in {x['test'] for x in stress} if all(x['passed'] for x in stress if x['test'] == n)),
                  environment=versions(), results=results)
    print(f'\n{PROTOCOL} v{VERSION}   {report["policy_file"]}')
    print(f'SHA-256  {report["policy_sha256"]}\n')
    for t in tests:
        clr = f"   clearance median {t['clearance_median_mm']} mm, min {t['clearance_min_mm']} mm" if 'clearance_median_mm' in t else ''
        print(f"  {t['engine']:8s} {t['test']:26s} falls {t['falls']}/{t['trials']}   min progress {t['min_progress']:.0%}{clr}")
    print(f'\nCore verdict: {report["verdict"].upper()}')
    print(f'\nStress tests (pass = at most {STRESS_MAX_FALLS} fall in 10, per engine): {report["stress_passed"]} of {len(STRESS_NAMES(stress))} passed')
    for x in stress: print(f"  {x['engine']:8s} {x['test']:26s} falls {x['falls']}/{x['trials']}   {'pass' if x['passed'] else 'FAIL' if x['passed'] is not None else 'not run'}")
    for f in fails[:12]: print('  - ' + f)
    if len(fails) > 12: print(f'  ... and {len(fails) - 12} more')
    if a.out: json.dump(report, open(a.out, 'w'), indent=1); print(f'\nReport written to {a.out}')
