"""MuJoCo side of the cross-check: the Go2 control loop rebuilt to match Isaac Lab exactly (observation layout, joint order, PD gains, torque-speed limit, 50 Hz policy on 200 Hz physics)."""
import numpy as np, mujoco, onnxruntime as ort, json, sys, argparse, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)

DT, DECIM = 0.005, 4
KP, KD, EFFORT, VLIM = 25.0, 0.5, 23.5, 30.0
ACT_SCALE = 0.25
# Isaac Lab Go2 joint order (grouped by joint type)
POLICY_JOINTS = [f"{l}_{j}_joint" for j in ("hip", "thigh", "calf") for l in ("FL", "FR", "RL", "RR")]
def default_pos(name):
    if "hip" in name: return 0.1 if name[1] == "L" else -0.1
    if "thigh" in name: return 0.8 if name[0] == "F" else 1.0
    return -1.5
DEFAULT = np.array([default_pos(n) for n in POLICY_JOINTS])

def build_model(friction=1.0, add_mass=0.0, matched=True):
    spec = mujoco.MjSpec.from_file(os.path.join(ROOT, "go2", "scene.xml"))
    spec.option.timestep = DT
    for g in spec.geoms:
        if g.name == "floor": g.friction[0] = friction
    for g in spec.geoms:
        if g.name in ("FL", "FR", "RL", "RR"): g.friction[0] = friction
    if matched:
        for j in spec.joints:
            if j.type == mujoco.mjtJoint.mjJNT_HINGE: j.damping[:] = 0.0; j.frictionloss = 0.0
    base = [b for b in spec.bodies if b.name == "base"][0]
    base.mass += add_mass
    m = spec.compile()
    return m

class Sim:
    def __init__(self, m):
        self.m, self.d = m, mujoco.MjData(m)
        self.qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in POLICY_JOINTS])
        self.vadr = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in POLICY_JOINTS])
        self.aid = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n.replace("_joint", "")) for n in POLICY_JOINTS])
        self.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # bodies whose ground contact ends an episode (Isaac Lab terminates on base contact)
        self.bad = {self.base}
    def reset(self, seed=None):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:3] = [0, 0, 0.4]; self.d.qpos[3:7] = [1, 0, 0, 0]
        self.d.qpos[self.qadr] = DEFAULT
        if seed is not None:  # seeded start: joint offsets +/-0.1 rad, small base velocity
            r = np.random.default_rng(1000 + seed)
            self.d.qpos[self.qadr] += r.uniform(-0.1, 0.1, 12)
            self.d.qvel[:2] = r.uniform(-0.2, 0.2, 2)
        mujoco.mj_forward(self.m, self.d)
        self.last = np.zeros(12)
    def R(self):
        return self.d.xmat[self.base].reshape(3, 3)
    def obs(self, cmd):
        R = self.R()
        lin = R.T @ self.d.qvel[:3]          # free joint linear vel is world frame
        ang = self.d.qvel[3:6].copy()        # free joint angular vel is body frame
        grav = R.T @ np.array([0, 0, -1.0])
        q = self.d.qpos[self.qadr] - DEFAULT
        qd = self.d.qvel[self.vadr]
        return np.concatenate([lin, ang, grav, cmd, q, qd, self.last]).astype(np.float32)
    def step(self, action):
        self.last = action
        target = DEFAULT + ACT_SCALE * action
        for _ in range(DECIM):
            q = self.d.qpos[self.qadr]; qd = self.d.qvel[self.vadr]
            tau = KP * (target - q) - KD * qd
            # Isaac Lab DCMotor torque-speed saturation
            hi = np.clip(EFFORT * (1 - qd / VLIM), 0, EFFORT); lo = np.clip(EFFORT * (-1 - qd / VLIM), -EFFORT, 0)
            self.d.ctrl[self.aid] = np.clip(tau, lo, hi)
            mujoco.mj_step(self.m, self.d)
    def fallen(self):
        if self.R()[2, 2] < 0.5: return True
        d = self.d
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            if self.floor in (g1, g2):
                other = g2 if g1 == self.floor else g1
                if self.m.geom_bodyid[other] in self.bad: return True
        return False
    def body_vel(self):
        R = self.R(); return R.T @ self.d.qvel[:3], self.d.qvel[5]

def run_episode(policy, m, cmd_fn, T, push_fn=None, settle=2.0, seed=None):
    sim = Sim(m); sim.reset(seed)
    errs_xy, errs_yaw, n = [], [], int(T / (DT * DECIM))
    for k in range(n):
        t = k * DT * DECIM
        cmd = np.array(cmd_fn(t), dtype=np.float32)
        a = policy.run(None, {"obs": sim.obs(cmd)[None]})[0][0]
        if push_fn: push_fn(t, sim)
        sim.step(a)
        if sim.fallen(): return dict(fell=True, t_fall=t, e_xy=np.nan, e_yaw=np.nan)
        if t > settle:
            v, wz = sim.body_vel()
            errs_xy.append(np.linalg.norm(v[:2] - cmd[:2])); errs_yaw.append(abs(wz - cmd[2]))
    return dict(fell=False, t_fall=None, e_xy=float(np.mean(errs_xy)), e_yaw=float(np.mean(errs_yaw)))

def make_pusher(seed, dv, period=3.0, start=3.0):
    rng = np.random.default_rng(seed); state = {"next": start}
    def push(t, sim):
        if t >= state["next"]:
            a = rng.uniform(0, 2 * np.pi); sim.d.qvel[0] += dv * np.cos(a); sim.d.qvel[1] += dv * np.sin(a)
            state["next"] += period
    return push

def scenarios(policy, seeds=10):
    out, m0 = {}, build_model()
    rep = lambda m, cmd, T, push=None: [run_episode(policy, m, cmd, T, push(k) if push else None, seed=k) for k in range(seeds)]
    for name, c in {"fwd_0.5": (0.5, 0, 0), "fwd_1.0": (1.0, 0, 0), "back_0.5": (-0.5, 0, 0),
                    "lat_0.5": (0, 0.5, 0), "yaw_1.0": (0, 0, 1.0), "stand": (0, 0, 0)}.items():
        out[f"track/{name}"] = rep(m0, lambda t, c=c: c, 10.0)
    for dv in (0.5, 1.0, 1.5):
        out[f"push/dv{dv}"] = rep(m0, lambda t: (0.5, 0, 0), 15.0, lambda k, dv=dv: make_pusher(k, dv))
    for mu in (0.4, 0.25):
        out[f"friction/{mu}"] = rep(build_model(friction=mu), lambda t: (0.8, 0, 0), 10.0)
    for kg in (3.0, 6.0):
        out[f"payload/+{kg}kg"] = rep(build_model(add_mass=kg), lambda t: (0.8, 0, 0), 10.0)
    def cmd_seq(seed):
        rng = np.random.default_rng(100 + seed); cs = rng.uniform(-1, 1, (4, 3)) * [1.0, 0.6, 1.0]
        return lambda t: cs[min(int(t // 5), 3)]
    out["random_cmds"] = [run_episode(policy, m0, cmd_seq(k), 20.0, seed=k) for k in range(seeds)]
    return out

def summarize(res):
    rows = {}
    for k, eps in res.items():
        falls = sum(e["fell"] for e in eps)
        exy = np.nanmean([e["e_xy"] for e in eps]) if falls < len(eps) else np.nan
        eyaw = np.nanmean([e["e_yaw"] for e in eps]) if falls < len(eps) else np.nan
        rows[k] = dict(falls=f"{falls}/{len(eps)}", e_xy=None if np.isnan(exy) else round(float(exy), 3), e_yaw=None if np.isnan(eyaw) else round(float(eyaw), 3))
    return rows
