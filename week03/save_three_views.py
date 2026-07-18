from __future__ import annotations

from pathlib import Path

import matplotlib

# 화면에 matplotlib 창을 띄우지 않고 파일로만 저장
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from envs.custom_lift import CustomLift


CAMERA_NAMES = [
    "frontview",
    "sideview",
    "robot0_eye_in_hand",
]

CAMERA_TITLES = {
    "frontview": "Front View",
    "sideview": "Side View",
    "robot0_eye_in_hand": "Eye in Hand",
}


def get_camera_names(env: CustomLift) -> list[str]:
    """현재 MuJoCo 모델에 등록된 카메라 이름을 반환한다."""

    camera_names: list[str] = []

    for camera_id in range(env.sim.model.ncam):
        name = env.sim.model.camera_id2name(camera_id)

        if name is not None:
            camera_names.append(name)

    return camera_names


def main() -> None:
    output_path = Path("custom_lift_3views.png")

    env = CustomLift(
        robots="Panda",

        # GUI 창은 띄우지 않는다.
        has_renderer=False,

        # 카메라 배열을 얻기 위해 반드시 True
        has_offscreen_renderer=True,
        use_camera_obs=True,

        # 사용할 세 카메라
        camera_names=CAMERA_NAMES,

        # 세 카메라 모두 같은 이미지 크기 사용
        camera_heights=256,
        camera_widths=256,

        # RGB만 사용
        camera_depths=False,

        use_object_obs=True,
        reward_shaping=True,

        # 매 reset마다 모델 전체를 다시 컴파일하지 않음
        hard_reset=False,

        horizon=200,
        control_freq=20,
    )

    try:
        obs = env.reset()

        # 환경에 실제로 등록된 카메라 확인
        available_cameras = get_camera_names(env)

        print("사용 가능한 카메라:")
        for camera_name in available_cameras:
            print(f"  - {camera_name}")

        # observation key 확인
        print("\n카메라 observation:")
        for camera_name in CAMERA_NAMES:
            image_key = f"{camera_name}_image"

            if image_key not in obs:
                raise KeyError(
                    f"관측값에 '{image_key}'가 없습니다.\n"
                    f"현재 observation key:\n{list(obs.keys())}"
                )

            print(
                f"  - {image_key}: "
                f"shape={obs[image_key].shape}, "
                f"dtype={obs[image_key].dtype}"
            )

        # 세 카메라 영상을 가로로 배치
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 5),
        )

        for axis, camera_name in zip(axes, CAMERA_NAMES):
            image_key = f"{camera_name}_image"
            image = obs[image_key]

            # MuJoCo와 matplotlib의 이미지 원점 차이 보정
            image = np.flipud(image)

            axis.imshow(image)
            axis.set_title(CAMERA_TITLES[camera_name])
            axis.axis("off")

        fig.suptitle(
            "CustomLift: Three Camera Views",
            fontsize=16,
        )

        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"\n3-view 이미지 저장 완료: {output_path.resolve()}")

    finally:
        env.close()


if __name__ == "__main__":
    main()