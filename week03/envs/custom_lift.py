from __future__ import annotations

import numpy as np

from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler


class CustomLift(Lift):
    """
    Lift 환경을 확장한 사용자 정의 환경.

    목표
    ----
    1. 빨간 상자와 파란 구를 테이블에 배치한다.
    2. reset마다 두 물체의 위치를 무작위로 바꾼다.
    3. 빨간 상자를 들면 보상을 준다.
    4. 파란 구를 들면 페널티를 준다.

    주의
    ----
    부모 Lift 클래스가 사용하는 self.cube 이름을 빨간 상자에 그대로 사용한다.
    그러면 Lift의 기존 grasp 판정, observable, visualization 기능을 재사용할 수 있다.
    """

    # 물체 중심이 테이블 상판보다 4 cm 이상 높으면 들었다고 판정한다.
    LIFT_HEIGHT_THRESHOLD = 0.04

    # 파란 구를 들었을 때 감점할 보상
    BLUE_BALL_PENALTY = 0.5

    def _load_model(self) -> None:
        """
        테이블, 로봇, 빨간 상자, 파란 구로 MuJoCo 모델을 구성한다.
        """

        # super()._load_model()을 호출하면 Lift의 기본 cube와 task가 생성된다.
        # 여기서는 그것을 피하고 Lift의 부모인 ManipulationEnv 단계부터 실행한다.
        ManipulationEnv._load_model(self)

        # 테이블 크기에 맞게 로봇 베이스 위치를 조정한다.
        base_xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(base_xpos)

        # 작업 공간인 테이블을 생성한다.
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # ---------------------------------------------------------
        # 물체 1: 빨간 상자
        # ---------------------------------------------------------
        # Lift의 기존 코드가 self.cube를 사용하므로 이름을 유지한다.
        # BoxObject의 size는 전체 길이가 아니라 각 축의 반길이다.
        self.cube = BoxObject(
            name="red_box",
            size=[0.022, 0.022, 0.022],
            rgba=[1.0, 0.0, 0.0, 1.0],
            density=500,
            friction=[1.0, 0.005, 0.0001],
            rng=self.rng,
        )

        # ---------------------------------------------------------
        # 물체 2: 파란 구
        # ---------------------------------------------------------
        # BallObject의 size는 반지름이다.
        self.blue_ball = BallObject(
            name="blue_ball",
            size=[0.025],
            rgba=[0.0, 0.0, 1.0, 1.0],
            density=500,
            friction=[1.0, 0.005, 0.001],
        )

        self.objects = [self.cube, self.blue_ball]

        # ---------------------------------------------------------
        # 초기 위치 랜덤화
        # ---------------------------------------------------------
        if self.placement_initializer is not None:
            # 외부에서 sampler를 전달받은 경우 두 물체를 등록한다.
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.objects)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="CustomObjectSampler",
                mujoco_objects=self.objects,
                x_range=[-0.10, 0.10],
                y_range=[-0.10, 0.10],
                rotation=None,
                rotation_axis="z",
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        # Arena + RobotModel + ObjectModel을 하나의 MJCF task로 합친다.
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )

    def _setup_references(self) -> None:
        """
        MuJoCo 배열에서 두 물체의 상태를 빠르게 읽기 위한 body ID를 저장한다.
        """

        # Lift._setup_references()가 self.cube_body_id를 설정한다.
        super()._setup_references()

        self.blue_ball_body_id = self.sim.model.body_name2id(
            self.blue_ball.root_body
        )

    def _check_success(self) -> bool:
        """
        빨간 상자의 중심 높이가 기준 높이보다 높으면 성공으로 판정한다.
        """

        red_box_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]

        return bool(
            red_box_height
            > table_height + self.LIFT_HEIGHT_THRESHOLD
        )

    def _blue_ball_is_lifted(self) -> bool:
        """
        파란 구가 기준 높이보다 올라갔는지 검사한다.
        """

        blue_ball_height = self.sim.data.body_xpos[
            self.blue_ball_body_id
        ][2]
        table_height = self.model.mujoco_arena.table_offset[2]

        return bool(
            blue_ball_height
            > table_height + self.LIFT_HEIGHT_THRESHOLD
        )

    def reward(self, action=None) -> float:
        """
        사용자 정의 보상 함수.

        Sparse reward:
            빨간 상자를 들면 +1.0

        Dense reward(reward_shaping=True):
            빨간 상자에 가까워질수록 최대 +0.4
            빨간 상자를 잡으면 +0.1

        Penalty:
            파란 구를 들면 -0.5
        """

        reward = 0.0

        # 목표 달성 보상
        if self._check_success():
            reward = 1.0

        # 목표를 달성하기 전 중간 보상
        elif self.reward_shaping:
            distance = self._gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.cube.root_body,
                target_type="body",
                return_distance=True,
            )

            # 거리가 0에 가까워질수록 약 0.4에 가까워진다.
            reaching_reward = 0.4 * (
                1.0 - np.tanh(10.0 * distance)
            )
            reward += reaching_reward

            # 그리퍼가 빨간 상자를 잡으면 추가 보상
            is_grasping_red_box = self._check_grasp(
                gripper=self.robots[0].gripper,
                object_geoms=self.cube,
            )

            if is_grasping_red_box:
                reward += 0.1

        # 방해 물체인 파란 구를 들면 페널티
        if self._blue_ball_is_lifted():
            reward -= self.BLUE_BALL_PENALTY

        # 최대 성공 보상이 1.0이므로 reward_scale을 그대로 곱한다.
        if self.reward_scale is not None:
            reward *= self.reward_scale

        return float(reward)