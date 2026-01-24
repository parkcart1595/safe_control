import numpy as np

from double_integrator2D import DoubleIntegrator2D
from backup_cbf_qp import BackupCBFQP

def fd_dhds(qp, pos, scenario, s, eps=1e-4):
    """Finite-difference approximation of ∂h/∂s at (pos, s)."""
    h_p, _, _ = qp._occlusion_barrier_smax_curved(pos, scenario, tau=s + eps)
    h_m, _, _ = qp._occlusion_barrier_smax_curved(pos, scenario, tau=s - eps)
    if h_p is None or h_m is None:
        return None
    return float(h_p - h_m) / (2.0 * eps)

def run_one_case():
    # --- minimal robot spec (match your code expectations) ---
    robot_spec = {
        "radius": 0.3,
        "a_max": 1.0,
        "v_max": 1.0,
        "w_max": 0.5,
        "sensing_range": 10.0,
        "v_adv_max_occ": 0.6,          # occlusion inflation speed bound
        "backup_cbf": {"T_horizon": 2.0, "dt_backup": 0.05, "alpha": 1.0},
        "debug_backup_qp": False,
    }

    dt = 0.05
    robot = DoubleIntegrator2D(dt, robot_spec)

    # Build QP object (we only use its occlusion barrier utilities)
    qp = BackupCBFQP(robot=robot, robot_spec=robot_spec, num_obs=10, kappa=10.0)

    # --- set a robot state and one obstacle to build a valid occlusion scenario ---
    robot_state = np.array([[0.0], [0.0], [0.0], [0.0]])  # at origin, stopped
    # obs format: [ox, oy, r, vx, vy, ..., ..., type]
    # type: 0 static, 1 dynamic (your code checks obs[7])
    obs = [3.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1]  # dynamic obstacle
    visible_obs, occ_scenarios = qp._filter_visible_and_build_occ(robot_state, [obs])

    assert len(occ_scenarios) > 0, "No occlusion scenario built. Try moving obstacle or increasing sensing_range."
    scenario = occ_scenarios[0]

    # --- pick a test position pos (2D) ---
    # Use a point somewhere in front of the robot; any point works for ∂h/∂s sign.
    pos = np.array([1.0, 0.2], dtype=float)

    # --- test multiple s values in [0, T] ---
    T = float(robot_spec["backup_cbf"]["T_horizon"])
    s_list = np.linspace(0.0, T, 9)

    print("\n=== Occlusion ∂h/∂s sign check ===")
    print("Interpretation: code's 'tau' argument here is lookahead s in [0,T].\n")

    rows = []
    for s in s_list:
        h_tilde, grad_pos, lam, dh_ds, _ = qp._occ_smax_details(pos, scenario, tau=s)
        dh_fd = fd_dhds(qp, pos, scenario, s, eps=1e-4)

        if h_tilde is None or dh_ds is None or dh_fd is None:
            rows.append([s, None, None, None, None])
            continue

        # In your code: time_term = -dh_ds
        time_term = -float(dh_ds)

        rows.append([
            float(s),
            float(h_tilde),
            float(dh_ds),     # analytic/code dh_ds = ∂h/∂s
            float(dh_fd),     # FD dh_ds
            float(time_term), # should be -∂h/∂s
        ])

    # Pretty print
    header = f"{'s':>6} | {'h_occ':>10} | {'dh_ds(code)':>12} | {'dh_ds(FD)':>10} | {'-dh_ds(time_term)':>16}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if r[1] is None:
            print(f"{r[0]:6.3f} | {'None':>10} | {'None':>12} | {'None':>10} | {'None':>16}")
        else:
            print(f"{r[0]:6.3f} | {r[1]:10.5f} | {r[2]:12.6f} | {r[3]:10.6f} | {r[4]:16.6f}")

    # Quantitative pass/fail
    diffs = []
    signs_ok = True
    for r in rows:
        if r[1] is None:
            continue
        s, h, dh_code, dh_fd, time_term = r
        diffs.append(abs(dh_code - dh_fd))

        # Expected: dh_ds = ∂h/∂s should be <= 0 if v_expand_vec >= 0
        if dh_code > 1e-6:
            signs_ok = False

        # Expected: time_term = -dh_ds should be >= 0
        if time_term < -1e-6:
            signs_ok = False

    if len(diffs) > 0:
        print("\nMax |dh_ds(code) - dh_ds(FD)| =", max(diffs))
        print("Sign checks passed? =", signs_ok)

        if signs_ok:
            print("\n✅ 결론: dh_ds는 ∂h/∂s로서 음수(또는 0)이고, time_term=-dh_ds는 양수 방향입니다.")
            print("   즉, 박헌국님 프레임(s=τ-t, ds/dt=-1)에서 time-term 부호는 현재 코드가 맞습니다.\n")
        else:
            print("\n❌ 부호가 어딘가 뒤집혀 있습니다.")
            print("   - v_expand_vec에 음수가 들어갔는지")
            print("   - h_k 정의를 -(R+v*s) 대신 +(R+v*s)로 썼는지")
            print("   - s 정의를 (t-τ)로 바꿔놓고 코드에 반영 안 했는지")
            print("   위를 우선 점검하세요.\n")
    else:
        print("\nCould not compute diffs (h_tilde None). Try different pos/obstacle.\n")

if __name__ == "__main__":
    run_one_case()
