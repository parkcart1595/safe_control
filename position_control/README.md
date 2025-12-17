# Backup CBF-QP (Occlusion-Aware)

This controller implements an occlusion-aware Backup CBF-QP for safety-critical navigation.
It is designed to run with the dynamic environment pipeline in `dynamic_env/main.py` and
currently ships with full support for the `DoubleIntegrator2D` model.

## Quick Start (DoubleIntegrator2D)

1) Install dependencies (see the repo root `SAFE_CONTROL/README.md`).
2) Run the dynamic environment entry point:

```bash
python dynamic_env/main.py
```

Make sure `single_agent_main` is called with the backup controller:

```python
single_agent_main(controller_type={'pos': 'backup_cbf_qp'})
```

By default, `dynamic_env/main.py` uses `DoubleIntegrator2D` for `model`. If you change
the model, see the "Model Requirements" section below.

## Robot Spec (Minimum)

The backup controller reads parameters from `robot_spec`. A minimal example:

```python
robot_spec = {
    "model": "DoubleIntegrator2D",
    "radius": 0.25,
    "a_max": 1.0,
    "sensing_range": 10.0,
    "backup_cbf": {
        "T_horizon": 3.0,
        "dt_backup": 0.05,
        "alpha": 2.0,
    },
}
```

## Solver Notes

`position_control/backup_cbf_qp.py` solves with GUROBI only. If GUROBI is not available,
`cvxpy` will raise a solver error. Make sure `gurobipy` is installed and the license is set
up for your environment.

## Where It Lives

- Controller: `position_control/backup_cbf_qp.py`
- Occlusion utilities: `utils/occlusion.py`
- Dynamic environment runner: `dynamic_env/main.py`

## Model Requirements (for other robots)

To run `backup_cbf_qp` with a new robot model, the robot class should provide:

- `u_dim` and `input_constraints(u_var)` for the QP
- `simulate_backup_trajectory(x0, T, dt, occlusion_scenarios=None)`
- `backup_input_occlusion(x, scenarios, t=None, ...)`
- `h_b_stop(x)` and `grad_h_b_stop(x)`
- `set_terminal_backup_context(occlusion_scenarios, T, kappa=None, rho_T=0.05)`
- `set_occ_barrier_fn(fn)` if you want occlusion visualization callbacks

The wrapper in `robots/robot.py` forwards these calls from the dynamic environment.

## Common Troubleshooting

- If you see an infeasible QP status, `tracking.py` handles it and draws the infeasible
  marker in the animation. Check solver logs for the exact status.
- If occlusion warnings appear, verify your robot implements `set_occ_barrier_fn`.
