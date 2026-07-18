from __future__ import annotations

import numpy as np

from envs.custom_lift import CustomLift


def main() -> None:
    env = CustomLift(
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        hard_reset=False,
        horizon=200,
        control_freq=20,
    )

    red_box_positions: list[np.ndarray] = []
    blue_ball_positions: list[np.ndarray] = []

    try:
        for reset_index in range(100):
            env.reset()

            # .copy()를 하지 않으면 MuJoCo 내부 배열을 계속 참조할 수 있다.
            red_box_pos = env.sim.data.body_xpos[
                env.cube_body_id
            ].copy()

            blue_ball_pos = env.sim.data.body_xpos[
                env.blue_ball_body_id
            ].copy()

            red_box_positions.append(red_box_pos)
            blue_ball_positions.append(blue_ball_pos)

            if reset_index < 5:
                print(
                    f"[reset {reset_index + 1:03d}] "
                    f"red_box={np.round(red_box_pos, 4)}, "
                    f"blue_ball={np.round(blue_ball_pos, 4)}"
                )

        red_box_array = np.asarray(red_box_positions)
        blue_ball_array = np.asarray(blue_ball_positions)

        # 소수점 넷째 자리까지 같은 위치를 동일한 위치로 간주한다.
        unique_red_positions = np.unique(
            np.round(red_box_array[:, :2], decimals=4),
            axis=0,
        )
        unique_blue_positions = np.unique(
            np.round(blue_ball_array[:, :2], decimals=4),
            axis=0,
        )

        distances = np.linalg.norm(
            red_box_array[:, :2] - blue_ball_array[:, :2],
            axis=1,
        )

        print("\n===== 랜덤화 검증 결과 =====")
        print(f"총 reset 횟수: {len(red_box_array)}")
        print(f"빨간 상자 고유 XY 위치 수: {len(unique_red_positions)}")
        print(f"파란 구 고유 XY 위치 수: {len(unique_blue_positions)}")

        print(
            "빨간 상자 XY 표준편차:",
            np.round(red_box_array[:, :2].std(axis=0), 4),
        )
        print(
            "파란 구 XY 표준편차:",
            np.round(blue_ball_array[:, :2].std(axis=0), 4),
        )
        print(
            "두 물체 사이 최소 XY 거리:",
            round(float(distances.min()), 4),
        )

        if len(unique_red_positions) <= 1:
            raise RuntimeError("빨간 상자의 위치가 랜덤화되지 않았습니다.")

        if len(unique_blue_positions) <= 1:
            raise RuntimeError("파란 구의 위치가 랜덤화되지 않았습니다.")

        print("\n랜덤화 검증 성공!")

    finally:
        env.close()


if __name__ == "__main__":
    main()