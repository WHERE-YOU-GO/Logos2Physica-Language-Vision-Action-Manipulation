from argparse import Namespace
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts.run_mini_task import (
    DuplicateAssignedPair,
    CubeTagPair,
    DuplicateCubeCandidate,
    DuplicatePosePlan,
    DuplicateTagCandidate,
    PresetAssignedPair,
    PresetLayout,
    PresetSlot,
    PosePlanSafetyAbort,
    _build_duplicate_pose_plan_refinement_child_command,
    _build_duplicate_pose_plan_robot_child_command,
    _build_checkpoint8_multi_subprocess_child_command,
    _checkpoint8_tcp_z_meets_minimum,
    _duplicate_pose_plan_from_preset_assignment,
    _execute_checkpoint8_pose_plan_grasp_place,
    _refine_duplicate_pose_plan_from_frame,
    _parse_args,
    _run_checkpoint8_pose_plan_child,
    _run_duplicate_pose_plan_child_subprocesses,
    duplicate_pose_plan_from_json_data,
    duplicate_pose_plan_to_json_data,
    filter_duplicate_cube_candidates,
    load_preset_place_layout_json,
    merge_duplicate_cube_candidates,
    parse_cube_tag_map,
    parse_candidate_merge_prompts,
    parse_duplicate_cube_tag_map,
    parse_preset_cube_counts,
    parse_preset_cube_slot_map,
    select_preset_assignment_subset,
    select_nearest_refined_candidate,
    select_duplicate_assignment_subset,
    solve_nearest_xy_assignment,
    validate_preset_layout_request,
)

import numpy as np


def _cube_candidate(instance_index, center, area_px=1000, score=0.0, color="blue", bbox_diag_m=0.030, max_extent_m=0.025):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(center, dtype=np.float64)
    return DuplicateCubeCandidate(
        cube_prompt=f"{color} cube",
        cube_color=color,
        instance_index=instance_index,
        component_label=instance_index,
        area_px=area_px,
        score=score,
        bbox_diag_m=bbox_diag_m,
        max_extent_m=max_extent_m,
        yaw_rad=0.0,
        T_robot_cube=pose,
        T_cam_cube=pose.copy(),
        center_robot=np.asarray(center, dtype=np.float64),
    )


def _tag_candidate(instance_index, center):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(center, dtype=np.float64)
    return DuplicateTagCandidate(
        tag_id=7,
        instance_index=instance_index,
        detection_index=instance_index - 1,
        decision_margin=100.0,
        hamming=0,
        T_robot_tag=pose,
        T_cam_tag=pose.copy(),
        center_robot=np.asarray(center, dtype=np.float64),
    )


class _FakeArm:
    def __init__(self, state=2, err_warn=(0, 0), position=None):
        self.state = state
        self.err_warn = list(err_warn)
        self.position = position or [174.97, 27.08, 31.60, 180.0, 0.0, 0.0]
        self.stop_gripper_calls = 0

    def get_state(self):
        return 0, self.state

    def get_err_warn_code(self):
        return 0, self.err_warn

    def get_position(self):
        return 0, self.position

    def get_servo_angle(self):
        return 0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def stop_lite6_gripper(self):
        self.stop_gripper_calls += 1
        return 0

    def disconnect(self):
        return 0


class _Completed:
    def __init__(self, returncode):
        self.returncode = returncode


def _pose_plan_command_args(**overrides):
    values = {
        "target_tag_size_m": 0.020,
        "_target_tag_size_m_arg": "0.020",
        "pose_plan_refine_radius_m": 0.060,
        "pose_plan_refine_tag_radius_m": 0.080,
        "candidate_x_min_m": 0.05,
        "candidate_x_max_m": 0.80,
        "candidate_y_min_m": -0.45,
        "candidate_y_max_m": 0.45,
        "candidate_z_min_m": 0.00,
        "candidate_z_max_m": 0.08,
        "candidate_min_area_px": 500,
        "red_candidate_min_area_px": 500,
        "green_candidate_min_area_px": 1200,
        "blue_candidate_min_area_px": 500,
        "candidate_min_extent_m": 0.012,
        "candidate_max_extent_m": 0.040,
        "candidate_merge_distance_m": 0.035,
        "table_z_m": 0.0,
        "point_cloud_scale": 1e-3,
        "min_after_grasp_z_mm": 100.0,
        "robot_ip": None,
        "robot_config": "config/robot.yaml",
        "continue_on_pair_failure": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _assigned_pair(index=1):
    cube = _cube_candidate(index, [0.100 + index * 0.010, 0.100, 0.0225], color="red")
    tag = _tag_candidate(index, [0.200 + index * 0.010, 0.100, 0.0])
    return DuplicateAssignedPair(
        execution_index=index,
        group_index=1,
        within_group_index=index,
        cube_prompt="red cube",
        tag_id=7,
        cube=cube,
        tag=tag,
        distance_m=0.010,
        T_robot_place=tag.T_robot_tag.copy(),
        T_cam_place=tag.T_cam_tag.copy(),
    )


def _preset_layout():
    return PresetLayout(
        name="hexagon_6_slots",
        frame="robot_base",
        slots={
            1: PresetSlot(slot_id=1, x=0.260, y=0.060, z=0.0225, yaw_deg=0.0),
            2: PresetSlot(slot_id=2, x=0.230, y=0.112, z=0.0225, yaw_deg=0.0),
            3: PresetSlot(slot_id=3, x=0.170, y=0.112, z=0.0225, yaw_deg=0.0),
            4: PresetSlot(slot_id=4, x=0.140, y=0.060, z=0.0225, yaw_deg=0.0),
            5: PresetSlot(slot_id=5, x=0.170, y=0.008, z=0.0225, yaw_deg=0.0),
            6: PresetSlot(slot_id=6, x=0.230, y=0.008, z=0.0225, yaw_deg=0.0),
        },
    )


def _preset_assigned_pair(index=1):
    cube = _cube_candidate(index, [0.100 + index * 0.010, 0.100, 0.0225], color="red")
    slot = PresetSlot(slot_id=index, x=0.200 + index * 0.010, y=0.100, z=0.0225, yaw_deg=0.0)
    tag = _tag_candidate(index, slot.center_robot)
    return PresetAssignedPair(
        execution_index=index,
        group_index=1,
        within_group_index=index,
        cube_prompt="red cube",
        cube=cube,
        slot=slot,
        tag=tag,
        distance_m=0.010,
        T_robot_place=tag.T_robot_tag.copy(),
        T_cam_place=tag.T_cam_tag.copy(),
        preset_use_slot_yaw=False,
    )


class CubeTagMapParserTest(unittest.TestCase):
    def test_accepts_three_pairs(self):
        self.assertEqual(
            parse_cube_tag_map("red cube:7,green cube:8,blue cube:9"),
            [
                ("red cube", 7),
                ("green cube", 8),
                ("blue cube", 9),
            ],
        )

    def test_rejects_invalid_values(self):
        invalid_values = [
            "",
            "red cube",
            ":7",
            "red cube:",
            "red cube:tag7",
            "red cube:7,red cube:8",
            "red cube:7,green cube:7",
            "red cube:7,",
        ]
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_cube_tag_map(value)

    def test_multi_subprocess_flags_parse(self):
        argv = [
            "run_mini_task.py",
            "--multi_place_to_tags",
            "--multi_subprocess",
            "--continue_on_pair_failure",
            "--auto_confirm",
            "--cube_tag_map",
            "red cube:6",
        ]
        with patch("sys.argv", argv):
            args = _parse_args()

        self.assertTrue(args.multi_subprocess)
        self.assertTrue(args.continue_on_pair_failure)
        self.assertTrue(args.auto_confirm)

    def test_default_candidate_merge_prompts_parse_to_green(self):
        with patch("sys.argv", ["run_mini_task.py"]):
            args = _parse_args()

        self.assertEqual(parse_candidate_merge_prompts(args.candidate_merge_prompts), ["green cube"])

    def test_candidate_merge_prompts_parser_kept_for_compatibility(self):
        self.assertEqual(parse_candidate_merge_prompts(""), [])
        self.assertEqual(parse_candidate_merge_prompts("green cube,red cube"), ["green cube", "red cube"])

    def test_builds_single_pair_child_command(self):
        args = Namespace(target_tag_size_m=0.02, _target_tag_size_m_arg="0.020")
        command = _build_checkpoint8_multi_subprocess_child_command(
            args,
            CubeTagPair(cube_prompt="blue cube", target_tag_id=7),
        )

        self.assertEqual(
            command,
            [
                "python",
                "scripts/run_mini_task.py",
                "--execution_backend",
                "checkpoint8_style",
                "--cube_prompt",
                "blue cube",
                "--place_to_tag",
                "--target_tag_id",
                "7",
                "--target_tag_size_m",
                "0.020",
                "--no_gui",
                "--auto_confirm",
            ],
        )
        self.assertIn("--auto_confirm", command)
        self.assertNotIn("--skip_home", command)
        self.assertNotIn("--no_final_home", command)
        self.assertNotIn("--multi_place_to_tags", command)
        self.assertNotIn("--multi_subprocess", command)
        self.assertNotIn("--continue_on_pair_failure", command)


class DuplicateCubeTagMapParserTest(unittest.TestCase):
    def test_accepts_duplicate_groups(self):
        self.assertEqual(
            parse_duplicate_cube_tag_map("red cube:6:2,green cube:8:2,blue cube:7:2"),
            [
                {"cube_prompt": "red cube", "tag_id": 6, "count": 2},
                {"cube_prompt": "green cube", "tag_id": 8, "count": 2},
                {"cube_prompt": "blue cube", "tag_id": 7, "count": 2},
            ],
        )

    def test_rejects_invalid_duplicate_values(self):
        invalid_values = [
            "",
            "red cube",
            "red cube:6",
            ":6:2",
            "red cube::2",
            "red cube:tag6:2",
            "red cube:6:",
            "red cube:6:two",
            "red cube:6:0",
            "red cube:6:-1",
            "red cube:6:11",
            "red cube:6:2,red cube:6:1",
            "red cube:6:2,",
        ]
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_duplicate_cube_tag_map(value)


class PresetLayoutParserTest(unittest.TestCase):
    def test_parses_preset_cube_counts(self):
        self.assertEqual(
            parse_preset_cube_counts("red cube:2,green cube:2,blue cube:2"),
            [
                {"cube_prompt": "red cube", "count": 2},
                {"cube_prompt": "green cube", "count": 2},
                {"cube_prompt": "blue cube", "count": 2},
            ],
        )

    def test_parses_preset_cube_slot_map(self):
        self.assertEqual(
            parse_preset_cube_slot_map("red cube:1,2;green cube:3,4;blue cube:5,6"),
            [
                {"cube_prompt": "red cube", "slot_ids": [1, 2]},
                {"cube_prompt": "green cube", "slot_ids": [3, 4]},
                {"cube_prompt": "blue cube", "slot_ids": [5, 6]},
            ],
        )

    def test_loads_and_validates_preset_layout_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            layout_path = Path(temp_dir) / "layout.json"
            layout_path.write_text(
                json.dumps(
                    {
                        "name": "test_layout",
                        "frame": "robot_base",
                        "slots": [
                            {"slot_id": 1, "x": 0.20, "y": 0.00, "z": 0.0225, "yaw_deg": 0.0},
                            {"slot_id": 2, "x": 0.23, "y": 0.05, "z": 0.0225, "yaw_deg": 0.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            layout = load_preset_place_layout_json(layout_path)

        self.assertEqual(layout.name, "test_layout")
        self.assertEqual(layout.frame, "robot_base")
        self.assertEqual(sorted(layout.slots), [1, 2])
        validate_preset_layout_request(
            layout=layout,
            cube_counts=parse_preset_cube_counts("red cube:2"),
            cube_slot_map=parse_preset_cube_slot_map("red cube:1,2"),
            slot_minimum_mm=np.array([120.0, -250.0, 15.0], dtype=np.float64),
            slot_maximum_mm=np.array([360.0, 250.0, 80.0], dtype=np.float64),
        )

    def test_rejects_duplicated_layout_slot_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            layout_path = Path(temp_dir) / "layout.json"
            layout_path.write_text(
                json.dumps(
                    {
                        "name": "bad_layout",
                        "frame": "robot_base",
                        "slots": [
                            {"slot_id": 1, "x": 0.20, "y": 0.00, "z": 0.0225},
                            {"slot_id": 1, "x": 0.23, "y": 0.05, "z": 0.0225},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_preset_place_layout_json(layout_path)

    def test_rejects_count_slot_mismatch(self):
        with self.assertRaises(ValueError):
            validate_preset_layout_request(
                layout=_preset_layout(),
                cube_counts=parse_preset_cube_counts("red cube:2"),
                cube_slot_map=parse_preset_cube_slot_map("red cube:1"),
                slot_minimum_mm=np.array([120.0, -250.0, 15.0], dtype=np.float64),
                slot_maximum_mm=np.array([360.0, 250.0, 80.0], dtype=np.float64),
            )

    def test_rejects_duplicated_slot_ids_in_map(self):
        with self.assertRaises(ValueError):
            validate_preset_layout_request(
                layout=_preset_layout(),
                cube_counts=parse_preset_cube_counts("red cube:1,green cube:1"),
                cube_slot_map=parse_preset_cube_slot_map("red cube:1;green cube:1"),
                slot_minimum_mm=np.array([120.0, -250.0, 15.0], dtype=np.float64),
                slot_maximum_mm=np.array([360.0, 250.0, 80.0], dtype=np.float64),
            )


class DuplicateAssignmentSolverTest(unittest.TestCase):
    def test_cross_assignment_is_globally_optimal(self):
        cube_centers = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        tag_centers = np.array([[9.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        assignment, distance_matrix, pair_distances = solve_nearest_xy_assignment(
            cube_centers,
            tag_centers,
            max_distance_m=20.0,
        )

        self.assertEqual(assignment, [1, 0])
        self.assertEqual(distance_matrix.shape, (2, 2))
        self.assertEqual(pair_distances, [1.0, 1.0])

    def test_rejects_distance_above_threshold(self):
        cube_centers = np.array([[0.0, 0.0, 0.0]])
        tag_centers = np.array([[0.50, 0.0, 0.0]])

        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                solve_nearest_xy_assignment(cube_centers, tag_centers, max_distance_m=0.30)

    def test_workspace_filter_rejects_out_of_bounds_and_tiny_area(self):
        candidates = [
            _cube_candidate(1, [-0.406, 0.279, 0.0225], area_px=206, score=0.0026),
            _cube_candidate(2, [0.2349, -0.1373, 0.0225], area_px=4193, score=0.0052),
        ]

        valid, rejected = filter_duplicate_cube_candidates(
            candidates,
            x_min_m=0.05,
            x_max_m=0.80,
            y_min_m=-0.45,
            y_max_m=0.45,
            z_min_m=0.00,
            z_max_m=0.08,
            min_area_px=500,
        )

        self.assertEqual([candidate.instance_index for candidate in valid], [2])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].candidate.instance_index, 1)
        self.assertIn("x=-406mm outside [50,800]mm", rejected[0].rejection_reasons)
        self.assertIn("area 206 < 500", rejected[0].rejection_reasons)

    def test_selection_chooses_real_blue_candidates_after_filtering(self):
        candidates = [
            _cube_candidate(1, [-0.406, 0.279, 0.0225], area_px=206, score=0.0026),
            _cube_candidate(2, [0.2349, -0.1373, 0.0225], area_px=4193, score=0.0052),
            _cube_candidate(3, [0.1882, -0.0709, 0.0225], area_px=3806, score=0.0107),
        ]
        tags = [
            _tag_candidate(1, [0.2360, -0.1380, 0.0]),
            _tag_candidate(2, [0.1890, -0.0715, 0.0]),
        ]
        valid, _rejected = filter_duplicate_cube_candidates(
            candidates,
            x_min_m=0.05,
            x_max_m=0.80,
            y_min_m=-0.45,
            y_max_m=0.45,
            z_min_m=0.00,
            z_max_m=0.08,
            min_area_px=500,
        )

        selection = select_duplicate_assignment_subset(
            valid,
            tags,
            count=2,
            max_distance_m=0.30,
        )

        self.assertEqual([candidate.instance_index for candidate in selection.selected_cubes], [2, 3])

    def test_assignment_solver_chooses_global_minimum_subset(self):
        cubes = [
            _cube_candidate(1, [0.700, 0.300, 0.0225], area_px=1000, score=0.0),
            _cube_candidate(2, [0.235, -0.137, 0.0225], area_px=1000, score=0.0),
            _cube_candidate(3, [0.188, -0.071, 0.0225], area_px=1000, score=0.0),
        ]
        tags = [
            _tag_candidate(1, [0.236, -0.138, 0.0]),
            _tag_candidate(2, [0.189, -0.072, 0.0]),
            _tag_candidate(3, [0.720, -0.300, 0.0]),
        ]

        selection = select_duplicate_assignment_subset(
            cubes,
            tags,
            count=2,
            max_distance_m=0.30,
        )

        self.assertEqual([candidate.instance_index for candidate in selection.selected_cubes], [2, 3])
        self.assertEqual([candidate.instance_index for candidate in selection.selected_tags], [1, 2])

    def test_filtering_can_leave_too_few_valid_candidates(self):
        candidates = [
            _cube_candidate(1, [-0.406, 0.279, 0.0225], area_px=206, score=0.0026),
            _cube_candidate(2, [0.2349, -0.1373, 0.0225], area_px=4193, score=0.0052),
        ]
        valid, _rejected = filter_duplicate_cube_candidates(
            candidates,
            x_min_m=0.05,
            x_max_m=0.80,
            y_min_m=-0.45,
            y_max_m=0.45,
            z_min_m=0.00,
            z_max_m=0.08,
            min_area_px=500,
        )

        with self.assertRaises(ValueError):
            select_duplicate_assignment_subset(
                valid,
                [_tag_candidate(1, [0.236, -0.138, 0.0]), _tag_candidate(2, [0.189, -0.072, 0.0])],
                count=2,
                max_distance_m=0.30,
            )

    def test_green_candidates_within_merge_distance_are_merged(self):
        candidates = [
            _cube_candidate(1, [0.200, 0.000, 0.0225], area_px=1500, color="green"),
            _cube_candidate(2, [0.225, 0.000, 0.0225], area_px=1700, color="green"),
            _cube_candidate(3, [0.350, 0.000, 0.0225], area_px=1800, color="green"),
        ]

        physical, merged = merge_duplicate_cube_candidates(candidates, merge_distance_m=0.035)

        self.assertEqual(len(physical), 2)
        self.assertEqual(len(merged), 2)
        self.assertEqual([candidate.instance_index for candidate in merged[0].merged_candidates], [1, 2])

    def test_largest_area_candidate_sources_yaw_but_center_is_weighted(self):
        candidates = [
            _cube_candidate(1, [0.200, 0.000, 0.0225], area_px=1000, score=0.001, color="green"),
            _cube_candidate(2, [0.220, 0.000, 0.0225], area_px=3000, score=0.020, color="green"),
        ]

        physical, merged = merge_duplicate_cube_candidates(candidates, merge_distance_m=0.035)

        self.assertEqual([candidate.instance_index for candidate in physical], [1])
        self.assertEqual(merged[0].physical_candidate.member_candidate_indices, (1, 2))
        self.assertAlmostEqual(physical[0].center_robot[0], 0.215)
        self.assertNotAlmostEqual(physical[0].center_robot[0], 0.220)

    def test_green_candidate_min_area_rejects_small_fragments(self):
        candidates = [
            _cube_candidate(1, [0.200, 0.000, 0.0225], area_px=1000, color="green"),
            _cube_candidate(2, [0.350, 0.000, 0.0225], area_px=1800, color="green"),
        ]

        valid, rejected = filter_duplicate_cube_candidates(
            candidates,
            x_min_m=0.05,
            x_max_m=0.80,
            y_min_m=-0.45,
            y_max_m=0.45,
            z_min_m=0.00,
            z_max_m=0.08,
            min_area_px=500,
            green_min_area_px=1200,
        )

        self.assertEqual([candidate.instance_index for candidate in valid], [2])
        self.assertEqual(rejected[0].candidate.instance_index, 1)
        self.assertIn("area 1000 < 1200", rejected[0].rejection_reasons)

    def test_red_like_fragments_cluster_into_two_physical_cubes(self):
        candidates = [
            _cube_candidate(1, [0.2932, -0.1476, 0.0225], area_px=1200, color="red"),
            _cube_candidate(2, [0.2195, -0.0088, 0.0225], area_px=1500, color="red"),
            _cube_candidate(3, [0.2288, -0.0161, 0.0225], area_px=1700, color="red"),
            _cube_candidate(4, [0.3057, -0.1500, 0.0225], area_px=1800, color="red"),
            _cube_candidate(5, [0.2276, 0.0005, 0.0225], area_px=1600, color="red"),
            _cube_candidate(6, [0.2968, -0.1358, 0.0225], area_px=1400, color="red"),
        ]

        physical, clusters = merge_duplicate_cube_candidates(candidates, merge_distance_m=0.035)

        self.assertEqual(len(physical), 2)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(physical[0].member_candidate_indices, (1, 4, 6))
        self.assertEqual(physical[1].member_candidate_indices, (2, 3, 5))

    def test_assignment_cannot_select_two_fragments_from_same_physical_cube(self):
        candidates = [
            _cube_candidate(1, [0.2932, -0.1476, 0.0225], area_px=1200, color="red"),
            _cube_candidate(2, [0.2195, -0.0088, 0.0225], area_px=1500, color="red"),
            _cube_candidate(3, [0.2288, -0.0161, 0.0225], area_px=1700, color="red"),
            _cube_candidate(4, [0.3057, -0.1500, 0.0225], area_px=1800, color="red"),
            _cube_candidate(5, [0.2276, 0.0005, 0.0225], area_px=1600, color="red"),
            _cube_candidate(6, [0.2968, -0.1358, 0.0225], area_px=1400, color="red"),
        ]
        physical, _clusters = merge_duplicate_cube_candidates(candidates, merge_distance_m=0.035)
        tags = [
            _tag_candidate(1, [0.298, -0.145, 0.0]),
            _tag_candidate(2, [0.225, -0.008, 0.0]),
        ]

        selection = select_duplicate_assignment_subset(physical, tags, count=2, max_distance_m=0.30)

        self.assertEqual(len(selection.selected_cubes), 2)
        self.assertEqual(selection.selected_cubes[0].member_candidate_indices, (1, 4, 6))
        self.assertEqual(selection.selected_cubes[1].member_candidate_indices, (2, 3, 5))
        selected_member_sets = [set(candidate.member_candidate_indices) for candidate in selection.selected_cubes]
        self.assertNotIn({2}, selected_member_sets)
        self.assertNotIn({3}, selected_member_sets)

    def test_assignment_uses_post_merge_physical_candidates(self):
        candidates = [
            _cube_candidate(1, [0.200, 0.000, 0.0225], area_px=1400, color="green"),
            _cube_candidate(2, [0.220, 0.000, 0.0225], area_px=2800, color="green"),
            _cube_candidate(3, [0.350, 0.000, 0.0225], area_px=2600, color="green"),
        ]
        physical, _merged = merge_duplicate_cube_candidates(candidates, merge_distance_m=0.035)
        tags = [
            _tag_candidate(1, [0.221, 0.000, 0.0]),
            _tag_candidate(2, [0.351, 0.000, 0.0]),
        ]

        selection = select_duplicate_assignment_subset(
            physical,
            tags,
            count=2,
            max_distance_m=0.30,
        )

        self.assertEqual([candidate.instance_index for candidate in selection.selected_cubes], [1, 2])
        self.assertEqual(selection.selected_cubes[0].member_candidate_indices, (1, 2))
        self.assertEqual(selection.selected_cubes[1].member_candidate_indices, (3,))


class PresetAssignmentSolverTest(unittest.TestCase):
    def test_preset_assignment_chooses_nearest_one_to_one_mapping(self):
        cubes = [
            _cube_candidate(1, [0.300, 0.000, 0.0225], area_px=1000, color="red"),
            _cube_candidate(2, [0.100, 0.000, 0.0225], area_px=1000, color="red"),
        ]
        slots = [
            PresetSlot(slot_id=1, x=0.105, y=0.000, z=0.0225),
            PresetSlot(slot_id=2, x=0.295, y=0.000, z=0.0225),
        ]

        selection = select_preset_assignment_subset(cubes, slots, count=2)

        self.assertEqual([cube.instance_index for cube in selection.selected_cubes], [1, 2])
        self.assertEqual(selection.slot_permutation, [1, 0])
        self.assertEqual([slots[index].slot_id for index in selection.slot_permutation], [2, 1])

    def test_preset_pose_plan_json_has_target_source(self):
        assignment = _preset_assigned_pair(1)

        plan = _duplicate_pose_plan_from_preset_assignment(assignment)
        data = duplicate_pose_plan_to_json_data(plan)
        loaded = duplicate_pose_plan_from_json_data(data)

        self.assertEqual(data["target_source"], "preset_slot")
        self.assertEqual(data["slot_id"], 1)
        self.assertEqual(data["preset_slot"]["slot_id"], 1)
        self.assertEqual(loaded.target_source, "preset_slot")
        self.assertEqual(loaded.slot_id, 1)
        self.assertEqual(loaded.preset_slot["slot_id"], 1)

    def test_refinement_skips_target_tag_for_preset_slot(self):
        args = _pose_plan_command_args()
        cube_pose = np.eye(4, dtype=np.float64)
        cube_pose[:3, 3] = [0.100, 0.100, 0.0225]
        slot = PresetSlot(slot_id=4, x=0.140, y=0.060, z=0.0225, yaw_deg=0.0)
        plan = DuplicatePosePlan(
            execution_index=1,
            cube_prompt="red cube",
            target_tag_id=4,
            cube_instance_index=1,
            tag_instance_index=4,
            T_robot_cube=cube_pose,
            T_robot_place=np.eye(4, dtype=np.float64),
            target_source="preset_slot",
            slot_id=4,
            preset_slot=slot.to_json_data(),
            preset_use_slot_yaw=False,
        )
        candidate = _cube_candidate(1, [0.102, 0.101, 0.0225], area_px=1000, color="red")

        with patch("scripts.run_mini_task._detect_duplicate_cube_candidates", return_value=[candidate]):
            with patch(
                "scripts.run_mini_task._detect_duplicate_target_tag_candidates",
                side_effect=AssertionError("preset refinement must not detect target tags"),
            ):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    refinement = _refine_duplicate_pose_plan_from_frame(
                        args=args,
                        plan=plan,
                        image=np.zeros((2, 2, 3), dtype=np.uint8),
                        point_cloud=np.zeros((2, 2, 3), dtype=np.float64),
                        camera_intrinsic=np.eye(3, dtype=np.float64),
                        T_cam_robot=np.eye(4, dtype=np.float64),
                        cube_size_m=0.025,
                    )

        self.assertIn("Preset slot target: skipping target AprilTag refinement.", output.getvalue())
        self.assertIsNone(refinement.refined_tag)
        self.assertAlmostEqual(refinement.T_robot_place[0, 3], slot.x)
        self.assertAlmostEqual(refinement.T_robot_place[1, 3], slot.y)
        self.assertAlmostEqual(refinement.T_robot_place[2, 3], candidate.T_robot_cube[2, 3])


class PosePlanRefinementSafetyTest(unittest.TestCase):
    def test_refinement_command_construction(self):
        args = _pose_plan_command_args()
        command = _build_duplicate_pose_plan_refinement_child_command(
            args,
            Path("logs/duplicate_pose_plans/raw_pair.json"),
            Path("logs/duplicate_pose_plans/refined_pair.json"),
        )

        self.assertIn("--refine_pose_plan_json", command)
        self.assertIn("logs/duplicate_pose_plans/raw_pair.json", command)
        self.assertIn("--refined_pose_plan_output_json", command)
        self.assertIn("logs/duplicate_pose_plans/refined_pair.json", command)
        self.assertIn("--target_tag_size_m", command)
        self.assertIn("0.020", command)
        self.assertNotIn("--execute_pose_plan_json", command)

    def test_robot_only_command_includes_no_pose_plan_refine(self):
        args = _pose_plan_command_args(robot_ip="192.168.1.158")
        command = _build_duplicate_pose_plan_robot_child_command(
            args,
            Path("logs/duplicate_pose_plans/refined_pair.json"),
        )

        self.assertIn("--execute_pose_plan_json", command)
        self.assertIn("logs/duplicate_pose_plans/refined_pair.json", command)
        self.assertIn("--no_pose_plan_refine", command)
        self.assertIn("--target_tag_size_m", command)
        self.assertIn("0.020", command)
        self.assertNotIn("--refine_pose_plan_json", command)

    def test_nearest_refined_candidate_selection_within_radius(self):
        candidates = [
            _cube_candidate(1, [0.100, 0.100, 0.0225], color="red"),
            _cube_candidate(2, [0.180, 0.100, 0.0225], color="red"),
        ]

        selected, distance_m = select_nearest_refined_candidate(
            candidates=candidates,
            planned_xy_m=np.array([0.108, 0.106], dtype=np.float64),
            max_distance_m=0.060,
            label="cube",
        )

        self.assertEqual(selected.instance_index, 1)
        self.assertAlmostEqual(distance_m, 0.010, places=6)

    def test_refinement_rejects_candidate_outside_radius(self):
        candidates = [_cube_candidate(1, [0.200, 0.000, 0.0225], color="red")]

        with self.assertRaises(ValueError):
            select_nearest_refined_candidate(
                candidates=candidates,
                planned_xy_m=np.array([0.000, 0.000], dtype=np.float64),
                max_distance_m=0.060,
                label="cube",
            )

    def test_post_grasp_z_check_rejects_low_retreat_height(self):
        tcp_pose = (174.97, 27.08, 31.60, 180.0, 0.0, 0.0)

        self.assertFalse(_checkpoint8_tcp_z_meets_minimum(tcp_pose, 100.0))

    def test_pose_plan_child_refuses_place_after_low_post_grasp_z(self):
        arm = _FakeArm(position=[174.97, 27.08, 31.60, 180.0, 0.0, 0.0])
        place_calls = []

        def grasp_cube(_arm, _pose):
            return None

        def place_cube(_arm, pose):
            place_calls.append(pose)

        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(PosePlanSafetyAbort):
                _execute_checkpoint8_pose_plan_grasp_place(
                    arm=arm,
                    execution_index=1,
                    T_robot_cube=np.eye(4, dtype=np.float64),
                    T_robot_place=np.eye(4, dtype=np.float64),
                    grasp_cube_fn=grasp_cube,
                    place_cube_fn=place_cube,
                    min_after_grasp_z_mm=100.0,
                )

        self.assertEqual(place_calls, [])
        self.assertEqual(arm.stop_gripper_calls, 1)
        self.assertIn(
            "Abnormal grasp/retreat height after grasp_cube; not calling place_cube.",
            output.getvalue(),
        )

    def test_robot_only_execution_path_does_not_call_refinement_or_zed(self):
        plan = DuplicatePosePlan(
            execution_index=1,
            cube_prompt="red cube",
            target_tag_id=7,
            cube_instance_index=1,
            tag_instance_index=1,
            T_robot_cube=np.eye(4, dtype=np.float64),
            T_robot_place=np.eye(4, dtype=np.float64),
        )
        arm = _FakeArm(position=[174.97, 27.08, 142.50, 180.0, 0.0, 0.0])
        place_calls = []

        def grasp_cube(_arm, _pose):
            return None

        def place_cube(_arm, pose):
            place_calls.append(pose)

        checkpoint1_module = types.SimpleNamespace(
            GRIPPER_LENGTH=67.0,
            grasp_cube=grasp_cube,
            place_cube=place_cube,
        )
        args = Namespace(
            skip_home=False,
            no_final_home=False,
            execute_pose_plan_json="refined_pair.json",
            no_pose_plan_refine=True,
            execute_pose_plan_refine_only=False,
            pose_plan_refine_before_execute=False,
            dry_run=False,
            min_after_grasp_z_mm=100.0,
        )

        with patch.dict(sys.modules, {"checkpoint1": checkpoint1_module}):
            with patch("scripts.run_mini_task._load_duplicate_pose_plan_json", return_value=plan):
                with patch(
                    "scripts.run_mini_task._run_checkpoint8_pose_plan_refinement",
                    side_effect=AssertionError("robot-only child must not open ZED"),
                ):
                    with patch("scripts.run_mini_task._checkpoint8_connect_arm", return_value=arm):
                        with patch("scripts.run_mini_task._checkpoint8_initialize_and_home"):
                            with patch("scripts.run_mini_task._checkpoint8_require_home_ready"):
                                with patch("scripts.run_mini_task._checkpoint8_move_home_required"):
                                    with contextlib.redirect_stdout(io.StringIO()):
                                        _run_checkpoint8_pose_plan_child(args, "192.168.1.158")

        self.assertEqual(len(place_calls), 1)

    def test_parent_aborts_if_refinement_child_fails_before_robot_motion(self):
        args = _pose_plan_command_args()
        assignment = _assigned_pair(1)
        calls = []

        def fake_run(command, cwd):
            calls.append(command)
            return _Completed(7)

        succeeded, failed = _run_duplicate_pose_plan_child_subprocesses(
            args=args,
            assignments=[assignment],
            plan_paths={1: Path("logs/duplicate_pose_plans/raw_pair.json")},
            subprocess_run_fn=fake_run,
        )

        self.assertEqual(succeeded, [])
        self.assertEqual(failed, [assignment])
        self.assertEqual(len(calls), 1)
        self.assertIn("--refine_pose_plan_json", calls[0])
        self.assertNotIn("--execute_pose_plan_json", calls[0])

    def test_parent_aborts_if_robot_child_fails(self):
        args = _pose_plan_command_args()
        assignment = _assigned_pair(1)
        calls = []
        returncodes = [0, 9]

        def fake_run(command, cwd):
            calls.append(command)
            return _Completed(returncodes.pop(0))

        succeeded, failed = _run_duplicate_pose_plan_child_subprocesses(
            args=args,
            assignments=[assignment],
            plan_paths={1: Path("logs/duplicate_pose_plans/raw_pair.json")},
            subprocess_run_fn=fake_run,
        )

        self.assertEqual(succeeded, [])
        self.assertEqual(failed, [assignment])
        self.assertEqual(len(calls), 2)
        self.assertIn("--refine_pose_plan_json", calls[0])
        self.assertIn("--execute_pose_plan_json", calls[1])
        self.assertIn("--no_pose_plan_refine", calls[1])


class DuplicatePosePlanJsonTest(unittest.TestCase):
    def test_pose_plan_json_round_trip(self):
        t_robot_cube = np.eye(4, dtype=np.float64)
        t_robot_cube[:3, 3] = [0.1, 0.2, 0.03]
        t_robot_place = np.eye(4, dtype=np.float64)
        t_robot_place[:3, 3] = [0.3, -0.2, 0.03]
        plan = DuplicatePosePlan(
            execution_index=3,
            cube_prompt="green cube",
            target_tag_id=8,
            cube_instance_index=2,
            tag_instance_index=1,
            T_robot_cube=t_robot_cube,
            T_robot_place=t_robot_place,
        )

        data = duplicate_pose_plan_to_json_data(plan)
        loaded = duplicate_pose_plan_from_json_data(data)

        self.assertEqual(loaded.execution_index, 3)
        self.assertEqual(loaded.cube_prompt, "green cube")
        self.assertEqual(loaded.target_tag_id, 8)
        self.assertEqual(loaded.cube_instance_index, 2)
        self.assertEqual(loaded.tag_instance_index, 1)
        np.testing.assert_allclose(loaded.T_robot_cube, t_robot_cube)
        np.testing.assert_allclose(loaded.T_robot_place, t_robot_place)
