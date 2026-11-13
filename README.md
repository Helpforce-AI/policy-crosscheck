# policy-crosscheck

Run a legged-robot policy trained in NVIDIA Isaac Lab, **unchanged**, in two other physics engines (MuJoCo and Genesis) and compare the verdicts.

Training metrics tell you how well a policy fits the engine it was trained in. They do not tell you whether it still works when the physics changes. This repo is the check we use at [Helpforce AI](https://www.helpforce.ai) before a policy goes anywhere near hardware.

## What we found

Two Unitree Go2 walking policies, both trained in Isaac Lab with the standard velocity-tracking task. The second adds one reward term that holds the body near 30 cm.

| Policy | Isaac Lab training survival |
|---|---|
| Standard reward | 99.8% |
| Helpforce Go2 v1 | 97.5% |

By the training numbers, the first policy looks better. Run unchanged in two other engines (falls out of 3 seeded trials, lower is better):

| Test | Standard, MuJoCo | Standard, Genesis | Helpforce v1, MuJoCo | Helpforce v1, Genesis |
|---|---|---|---|---|
| Walk 0.5 m/s | 3 | 3 | 0 | 0 |
| Walk 1.0 m/s | 2 | 0 | 0 | 0 |
| 6 kg payload at 0.8 m/s | 3 | 3 | 0 | 0 |
| Friction 0.25 at 0.8 m/s | 3 | 2 | 0 | 0 |

The standard-reward policy learned a deep crouch: at 1.0 m/s it walks with a median of about **17 mm** between its body and the floor. Whether that counts as touching the ground depends on each engine's contact model, so the engines disagree. Helpforce Go2 v1 keeps about **195 mm** and every engine gives it the same verdict.

**When engines disagree about a policy, the policy is relying on engine-specific physics.** Disagreement is a signal to investigate, not noise to average away.

Full write-up: [methodology and scorecard](https://www.helpforce.ai/simulator/methodology). Try both policies live in your browser: [Helpforce Simulator](https://www.helpforce.ai/simulator).

## Run it

```bash
pip install -r requirements.txt            # MuJoCo and ONNX Runtime
pip install genesis-world                  # optional, for --engine genesis (CPU is fine)

python crosscheck.py policies/helpforce_go2_v1.onnx
python crosscheck.py policies/standard_reward.onnx --engine both --out results/mine.json
```

MuJoCo takes a few seconds per policy. Genesis on CPU takes about a minute.

### Your own policy

Any Go2 policy exported from Isaac Lab's flat velocity task should work as it is:

- **Input** `obs`, 48 values: base linear velocity (body frame), base angular velocity, gravity direction (body frame), command (vx, vy, yaw rate), 12 joint positions relative to default, 12 joint velocities, 12 previous actions.
- **Output** 12 actions, scaled by 0.25 and added to the default pose as joint targets.
- **Joint order** is Isaac Lab's: all hips, then all thighs, then all calves (FL, FR, RL, RR). The MuJoCo model orders joints leg by leg; `mujoco_engine.py` maps between them.
- **Control:** PD with stiffness 25 and damping 0.5, torque limit 23.5 N·m falling with joint speed (Isaac Lab's DC motor model), policy at 50 Hz on 200 Hz physics.

If your observation layout, gains or timing differ, change the constants at the top of `crosscheck/mujoco_engine.py`. Most transfer bugs are interface mismatches, not physics.

## Check against the Helpforce Validated protocol

`certify.py` scores a policy against [protocol HV-LOCO-1 v1.1](https://www.helpforce.ai/validated/protocol), in two tiers.

**Core** (must pass): 4 tests x 10 seeds x 2 engines, 80 trials.

- **C1** no falls in any trial
- **C2** every trial covers at least 70% of the commanded distance
- **C3** in MuJoCo, median body clearance at least 50 mm in every test, and no trial below 20 mm

**Stress** (graded, added in v1.1): sideways pushes of 1.0 m/s, a 10 kg payload, 20 ms action latency and Isaac Lab's default sensor noise. Another 80 trials. A stress test passes when no more than 1 of 10 trials falls in each engine.

```bash
python certify.py policies/helpforce_go2_v1.onnx --out report.json   # about 6 minutes on a CPU
python crosscheck.py policies/your_policy.onnx --engine both --stress  # the raw table
```

Results for the two included policies (falls out of 10, MuJoCo / Genesis):

| Test | Standard reward | Helpforce Go2 v1 | Helpforce Go2 v2 |
|---|---|---|---|
| Core, all four tests | 36 / 27 of 40 | 0 / 0 of 40 | 0 / 0 of 40 |
| Push 1.0 m/s | 10 / 7 | 6 / 3 | 0 / 0 |
| 10 kg payload | 10 / 10 | 10 / 6 | 0 / 0 |
| 20 ms latency | 4 / 2 | 0 / 0 | 0 / 0 |
| Sensor noise | 9 / 1 | 0 / 0 | 0 / 0 |
| Body clearance, MuJoCo median per test | 17 mm at 1.0 m/s | 150 to 196 mm | 212 to 236 mm |

Helpforce Go2 v1 meets the core protocol and passes 2 of 4 stress tests ([certificate HV-2026-0001](https://www.helpforce.ai/validated/HV-2026-0001)). It is weak against strong pushes and heavy loads, and we publish that. The standard-reward policy fails the core and all four stress tests.

**Helpforce Go2 v2** is v1 fine-tuned for 500 more iterations in Isaac Lab with random pushes (up to 1.0 m/s) and added body mass (-1 to +12 kg) switched on. It meets the core protocol and passes all 4 stress tests in both engines. Read that result with care: v2 was trained on the same kinds of disturbance the stress tier uses, so passing them is expected. What the run shows is that the fix carries over unchanged into MuJoCo and Genesis, and that core walking did not get worse. The exact Isaac Lab training config and code change for v1 and v2 are in `training/`.

All runs are in `results/`.

The script tells you whether a file meets the protocol. Certificates are issued only by Helpforce AI after rerunning the check; see [Helpforce Validated](https://www.helpforce.ai/validated). This is simulation evidence, not a safety certification.

## How a trial works

- Each trial is a 6 s walk at a commanded forward speed, from a seeded start (joint offsets of up to 0.1 rad and a small base velocity). Seed `k` gives the same start in both engines.
- A trial **fails** if the body touches the ground or tilts past 60 degrees, the same rule Isaac Lab uses to end an episode.
- In MuJoCo the script also measures the gap between the underside of the body and the floor, for trials that walk the full 6 s.

## Files

```
crosscheck.py               the four tests, both engines, results table and JSON
certify.py                  scores a policy against protocol HV-LOCO-1 v1.1 (core + stress tiers, file hash)
crosscheck/mujoco_engine.py MuJoCo control loop matched to Isaac Lab
crosscheck/genesis_engine.py Genesis control loop, same constants and fall rule
policies/                   the Go2 policies above (ONNX): standard reward, Helpforce v1, Helpforce v2
results/                    the runs behind the tables above
training/                   Isaac Lab training config (env.yaml, agent.yaml) and code change (IsaacLab.diff) for v1 and v2
go2/                        Unitree Go2 model from MuJoCo Menagerie (BSD-3-Clause, see go2/LICENSE)
```

## Caveats

- Simulation results only. No policy here has been tested on a physical Go2.
- Genesis runs with its default contact settings, which leave out the elliptic friction cone and the torsional and rolling friction declared in the model file. That is how Genesis behaves out of the box, and it is part of what is being tested.
- Three seeds per engine is a small sample. Use `--seeds 10` for tighter numbers.
- Isaac Lab itself is not run here. The Isaac numbers above are from training.

## Want this for your robot?

We run policies for the Go2 and other robots through checks like this, plus stress sweeps and a comparison against a reference, and send a scored report. [Book a 20-minute validation call](https://www.helpforce.ai/contact).

## License

Code and policies: Apache-2.0. The `go2/` model is from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) under its own BSD-3-Clause license.

If you use this in your work, please cite:

```
@misc{helpforce2026crosscheck,
  title  = {policy-crosscheck: testing Isaac Lab locomotion policies across physics engines},
  author = {Asghar, Usman Ali and {Helpforce AI}},
  year   = {2026},
  url    = {https://github.com/Helpforce-AI/policy-crosscheck}
}
```
