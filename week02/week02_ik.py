import csv
import time
import numpy as np


# ============================================================
# Part A: Numerical IK with Damped Least Squares
# ============================================================

L1 = 1.0
L2 = 1.0


def fk(q):
    """
    2-link planar robot forward kinematics.

    q: np.array([q1, q2])
    return: np.array([x, y])
    """
    q1, q2 = q

    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)

    return np.array([x, y])


def jacobian(q):
    """
    Jacobian of 2-link planar robot.

    J = d(x, y) / d(q1, q2)
    """
    q1, q2 = q

    j11 = -L1 * np.sin(q1) - L2 * np.sin(q1 + q2)
    j12 = -L2 * np.sin(q1 + q2)

    j21 = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    j22 = L2 * np.cos(q1 + q2)

    J = np.array([
        [j11, j12],
        [j21, j22],
    ])

    return J


def dls_pseudoinverse(J, damping=0.05):
    """
    Damped Least Squares pseudoinverse.

    J_dls = J.T @ inv(J @ J.T + lambda^2 I)
    """
    m = J.shape[0]
    I = np.eye(m)

    A = J @ J.T + (damping ** 2) * I

    # np.linalg.inv(A)보다 solve가 수치적으로 더 안정적임
    J_dls = J.T @ np.linalg.solve(A, I)

    return J_dls


def numerical_ik(
    target,
    q_init,
    max_iter=100,
    tolerance=1e-4,
    damping=0.05,
    singular_threshold=1e-6,
):
    """
    Numerical IK using Newton-Raphson style update with DLS.

    q_new = q - J^+ (fk(q) - target)
          = q + J^+ (target - fk(q))
    """
    q = q_init.astype(float).copy()

    for i in range(max_iter):
        current = fk(q)
        error = target - current
        error_norm = np.linalg.norm(error)

        print(
            f"iter={i:03d}, "
            f"q={q}, "
            f"current={current}, "
            f"error={error}, "
            f"error_norm={error_norm:.6f}"
        )

        if error_norm < tolerance:
            print(f"Converged at iteration {i}")
            return q

        J = jacobian(q)

        det_jjt = np.linalg.det(J @ J.T)

        if det_jjt < singular_threshold:
            print(
                f"[Warning] Near singularity: "
                f"det(JJ^T) = {det_jjt:.3e}"
            )

        J_dls = dls_pseudoinverse(J, damping=damping)

        dq = J_dls @ error

        q = q + dq

    print("Failed to converge within max_iter")
    return q


def run_part_a():
    print("\n==============================")
    print("Part A: Numerical IK")
    print("==============================")

    target = np.array([1.2, 0.8])
    q_init = np.array([0.1, 0.1])

    q_solution = numerical_ik(
        target=target,
        q_init=q_init,
        max_iter=100,
        tolerance=1e-4,
        damping=0.05,
        singular_threshold=1e-6,
    )

    print("\nFinal result")
    print("target:", target)
    print("q_solution:", q_solution)
    print("fk(q_solution):", fk(q_solution))
    print("final_error:", target - fk(q_solution))


# ============================================================
# Part B: RoboSuite Lift random action rollout
# ============================================================

def run_robosuite_lift(
    csv_path="robosuite_lift_log.csv",
    horizon=200,
    render=True,
):
    """
    Run one episode in RoboSuite Lift environment with random actions.

    Save joint positions, end-effector position, and reward to CSV.
    """
    import robosuite as suite

    print("\n==============================")
    print("Part B: RoboSuite Lift")
    print("==============================")

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=horizon,
    )

    obs = env.reset()

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "step",
            "joint_pos",
            "eef_pos",
            "reward",
            "done",
        ])

        for step in range(horizon):
            action = np.random.uniform(
                low=-1.0,
                high=1.0,
                size=env.action_dim,
            )

            obs, reward, done, info = env.step(action)

            joint_pos = obs.get("robot0_joint_pos", None)
            eef_pos = obs.get("robot0_eef_pos", None)

            writer.writerow([
                step,
                joint_pos.tolist() if joint_pos is not None else None,
                eef_pos.tolist() if eef_pos is not None else None,
                reward,
                done,
            ])

            if render:
                env.render()
                time.sleep(0.03)

            if done:
                break

        if render:
            input("Press Enter to close the RoboSuite window...")

        env.close()
        print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    run_part_a()

    # 스크린샷이 필요하면 render=True로 실행
    # Mac에서 render=True가 안 되면 mjpython week02_ik.py로 실행
    run_robosuite_lift(
        csv_path="robosuite_lift_log.csv",
        horizon=1000,
        render=True,
    )