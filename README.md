# Logos2Physica — Language–Vision–Action Manipulation

A modular **language-conditioned tabletop manipulation** stack for a Lite6-class arm: one **natural-language command** drives **scene grounding**, **RGB-D geometry**, **grasp / place planning**, **Cartesian execution**, **gripper control**, and **post-action verification**. The system is organized as a transparent, debuggable pipeline rather than an end-to-end black box:

**sense → parse → ground → project → plan → grasp → verify → place**

The repository is designed for **three practical operating regimes**:

1. **Replay / fake-robot validation** (fully safe, no hardware required)
2. **Real perception + dry planning** (real camera / detector, no robot motion)
3. **Staged real-robot bring-up** (true Lite6 execution, introduced incrementally)

---

## 1. What this repository does

| Layer | Role |
|------|------|
| **Language** | Converts a short English instruction into a structured `ParsedCommand` (`source`, optional `target`, relation, action type). |
| **Perception** | Uses an open-vocabulary detector backend and RGB-D projection to build 3D `SceneObject`s in the robot base frame. |
| **Planning** | Produces grasp pose, place pose, and Cartesian waypoint segments for pick-and-place. |
| **Control & Safety** | Applies workspace guardrails, then executes motion and gripper commands through a robot adapter. |
| **Verification** | Checks post-grasp and post-place outcomes, enabling bounded retries if desired. |

The primary user-facing task is simple:

> **Give the robot a short English command so it grasps the correct object, and optionally places it relative to another object or region.**

---

## 2. Current validated operating modes

At the current stage of the project, the following modes are the most important:

### Mode A — Replay scene + demo detector + fake robot
This is the safest and most reproducible mode. It uses:

- a saved RGB-D replay scene from `data/scenes/scene_01`
- the lightweight `demo` backend (`ColorBlockDemoDetector`)
- the `FakeLite6Adapter` instead of the real robot

This mode is the **recommended default** for:
- software validation
- demos without hardware risk
- debugging parsing / scene-state / planning / FSM transitions

### Mode B — Replay scene + demo detector + planning only
This mode validates scene understanding and motion planning without executing the full FSM.

### Mode C — Real robot staged bring-up
This is the hardware path. It should be introduced in stages:
1. connect to the robot
2. read current robot state
3. test minimal motion
4. test gripper open / close
5. test empty-space pregrasp / grasp / retreat
6. only then move to real object manipulation

### Mode D — Real detector / real camera / full closed loop
This is the final experimental mode, but it is **not the first recommended hardware step**. It depends on:
- real calibrated RGB-D input
- real detector weights and model paths
- correct `T_base_cam`
- validated workspace limits
- successful staged robot bring-up

---

## 3. End-to-end pipeline (FSM)

The orchestrator is `fsm/main_fsm.py`, implemented as `Prompt2PoseFSM`. A full execution run proceeds through these conceptual stages:

1. **Parse command**  
   Convert the user’s natural-language instruction into a `ParsedCommand`. The current primary path is regex-based parsing. Optional LLM-based parsing can be added when needed.

2. **Sense scene**  
   Acquire a frame (real camera or replay provider), detect relevant objects, and construct a `SceneState`.

3. **Resolve targets**  
   Match the linguistic description (e.g., *red cube*, *blue block*) to concrete scene object IDs.

4. **Plan**  
   Build a pick-and-place plan including:
   - `pregrasp_pose`
   - `grasp_pose`
   - `retreat_pose`
   - `place_pose`
   - waypoint sequence

5. **Safety check**  
   Validate workspace bounds, waypoint heights, and overall motion legality.

6. **Execute pick**  
   Run approach, grasp, gripper close, and retreat.

7. **Verify grasp**  
   Check whether grasping appears successful.

8. **Execute place**  
   If grasp is valid, approach the place location, release, and retreat.

9. **Verify place**  
   Check whether the object was placed as intended.

This decomposition is deliberate: the project separates **semantic understanding**, **physical grounding**, **planning**, **control**, and **verification** so that each layer can be tested independently.

---

## 4. Natural-language command format

The repository currently supports **short, structured English commands**, rather than unrestricted conversation.

Supported verbs include:

- `pick up`
- `pick`
- `grab`
- `put`
- `place`
- `move`

### 4.1 Pick-only commands

Use one of:

```text
pick up <object description>
pick <object description>
grab <object description>
