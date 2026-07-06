# week01_fk.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import roboticstoolbox as rtb


def load_ur5():
    """
    Robotics Toolbox 버전에 따라 UR5 로드 경로가 다를 수 있어서
    URDF.UR5()를 먼저 시도하고, 실패하면 models.UR5()를 시도한다.
    """
    try:
        return rtb.models.URDF.UR5()
    except AttributeError:
        return rtb.models.UR5()


def make_joint_trajectory(n=50):
    """
    임의 관절각 50개 시퀀스 생성.
    shape: (50, 6)
    """

    q_start = np.array([0, 0, 0, 0, 0, 0], dtype=float)

    q_end = np.array([
        np.pi / 3,
        -np.pi / 4,
        np.pi / 3,
        -np.pi / 6,
        np.pi / 4,
        np.pi / 2
    ], dtype=float)

    qs = np.linspace(q_start, q_end, n)
    return qs


def compute_fk_path(robot, qs):
    """
    각 관절각 q에 대해 FK를 계산하고,
    엔드이펙터의 4x4 변환행렬과 xyz 위치를 저장한다.
    """

    transforms = []
    positions = []

    for q in qs:
        T = robot.fkine(q)

        # T는 spatialmath.SE3 객체이므로 .A로 4x4 numpy matrix를 얻는다.
        T_mat = T.A

        # 위치는 4x4 행렬의 마지막 열 중 위 3개 값
        xyz = T_mat[:3, 3]

        transforms.append(T_mat)
        positions.append(xyz)

    transforms = np.array(transforms)
    positions = np.array(positions)

    return transforms, positions


def save_path_gif(positions, filename="ur5_path.gif"):
    """
    엔드이펙터 xyz 경로를 3D 애니메이션 gif로 저장한다.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    margin = 0.1

    ax.set_xlim(np.min(x) - margin, np.max(x) + margin)
    ax.set_ylim(np.min(y) - margin, np.max(y) + margin)
    ax.set_zlim(np.min(z) - margin, np.max(z) + margin)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("UR5 End-Effector Path")

    line, = ax.plot([], [], [], marker="o")
    point, = ax.plot([], [], [], marker="o")

    def update(frame):
        line.set_data(x[:frame + 1], y[:frame + 1])
        line.set_3d_properties(z[:frame + 1])

        point.set_data([x[frame]], [y[frame]])
        point.set_3d_properties([z[frame]])

        return line, point

    ani = FuncAnimation(
        fig,
        update,
        frames=len(positions),
        interval=100,
        blit=False
    )

    ani.save(filename, writer=PillowWriter(fps=10))
    plt.close(fig)


def compare_single_joint_motion(robot, filename="single_joint_comparison.png"):
    """
    특정 관절 하나만 움직였을 때 엔드이펙터 궤적 비교.
    결과는 png로 저장한다.
    """

    n = 100
    angles = np.linspace(-np.pi, np.pi, n)

    q_base = np.array([0, -np.pi / 4, np.pi / 3, 0, np.pi / 4, 0], dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for joint_idx in range(6):
        qs = np.tile(q_base, (n, 1))
        qs[:, joint_idx] = angles

        _, positions = compute_fk_path(robot, qs)

        ax.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            label=f"Joint {joint_idx + 1}"
        )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("End-Effector Trajectory by Single Joint Motion")
    ax.legend()

    plt.savefig(filename, dpi=200)
    plt.close(fig)


def dh_transform(a, alpha, d, theta):
    """
    Standard DH 변환행렬.
    """

    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T = np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1]
    ], dtype=float)

    return T


def manual_ur5_fk(q):
    """
    UR5 DH 파라미터를 이용한 수동 FK 계산.
    Robotics Toolbox 결과와 비교용으로 사용한다.
    """

    ur5_dh = [
        # a, alpha, d
        [0.0,       np.pi / 2, 0.089159],
        [-0.425,    0.0,       0.0],
        [-0.39225,  0.0,       0.0],
        [0.0,       np.pi / 2, 0.10915],
        [0.0,      -np.pi / 2, 0.09465],
        [0.0,       0.0,       0.0823],
    ]

    T = np.eye(4)

    for i in range(6):
        a, alpha, d = ur5_dh[i]
        theta = q[i]
        T = T @ dh_transform(a, alpha, d, theta)

    return T


def main():
    robot = load_ur5()

    print(robot)

    qs = make_joint_trajectory(n=50)

    transforms, positions = compute_fk_path(robot, qs)

    print("첫 번째 pose의 4x4 변환행렬:")
    print(transforms[0])

    print("\n마지막 pose의 4x4 변환행렬:")
    print(transforms[-1])

    save_path_gif(positions, filename="ur5_path.gif")
    compare_single_joint_motion(robot, filename="single_joint_comparison.png")

    # 수동 DH FK와 비교
    q_test = qs[-1]
    T_manual = manual_ur5_fk(q_test)

    print("\nManual DH FK 결과:")
    print(T_manual)

    print("\nRobotics Toolbox FK 결과:")
    print(transforms[-1])

    print("\n저장 완료:")
    print("- ur5_path.gif")
    print("- single_joint_comparison.png")


if __name__ == "__main__":
    main()