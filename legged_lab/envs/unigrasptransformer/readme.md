# UniGraspTransformer RL Notes (Kid-Friendly Edition)

Imagine the robot hand is a kid stacking blocks. It needs to **see**, **move**, and **score points**. Below is how both our Isaac Lab port and the original Microsoft Research task teach the kid.

## Observation Space (What the hand "sees")

- **Big picture:** Every step still emits 400 numbers, but the first five sections now match **exactly** the five groups from Table 1 in the UniGraspTransformer paper (167 + 24 + 16 + 36 + 29 = 272). We tack on the 128-D visual feature tail afterward to stay layout-compatible with the original Isaac Gym task.
- **Proprioception (0–166):** 22 finger joints × (position, velocity, pretend force) plus fingertip pose/velocity/force/torque and the palm pose in the object frame, all concatenated into a 167-D block.
- **Previous action (167–190):** The last palm translation/rotation deltas and finger joint deltas, rotated into the current object frame so PPO and the offline distillation data see object-centric control history.
- **Object state (191–206):** Object center, quaternion, linear velocity, angular velocity, and goal offset (3 + 4 + 3 + 3 + 3 = 16). We still express these values in the object frame, so position/orientation entries are zeroed when the hand already works in that frame.
- **Hand–object distance (207–242):** Minimum distances from the same 36 canonical palm/finger points used upstream (19 synthetic offsets + 17 joint centers) to the object point cloud. Zeros will appear if the cloud is missing, but the slot remains reserved.
- **Time (243–271):** Current normalized timestep plus a 28-D sinusoidal encoding copied from the upstream `progress_buf` helper so training curves line up with Table 1 (29 total values).
- **Object visual add-on (272–399):** Static/dynamic PointNet embeddings in the official code. In our port we fill it with centered FPS point-cloud slices when available and otherwise leave it zero, which keeps the total vector length at 400 even if those features are unused.

## Action Space (How the hand moves)

- **Size:** 24 numbers every step.
- **Parts:**
  1. **Palm translation (3)** and **palm rotation (3):** These steer the whole hand. We clamp them to [-1,1], scale them, and convert into object coordinates (same as upstream PPO).
  2. **Finger joint targets (18):** Each finger joint gets a delta added to its default angle. PPO learns to curl or uncurl to hug the object.
- **Delay buffer:** We keep the same action-delay option as upstream; it lets us pretend the kid’s brain needs a moment before moving.

## Reward Decomposition (How points are scored)

The reward function lives in `legged_lab/mdp/rewards_unigrasptransformer.py` and mirrors `dexgrasp/utils/reward` pieces. Think of two phases:

1. **Getting ready (“init” reward):**
   - Stay close to the desired starting finger pose.
   - Bring the palm near the object’s point cloud.
   - Point the palm’s main axis toward the object’s PCA axis.
   - Explore random points on the object surface (encourages reaching around).
2. **Grasping (“grasp” reward):**
   - Keep palm, fingers, joints, and hand body parts close to the object cloud.
   - Move the object toward the goal height (goal distance + “hand up” bonuses).
   - Maintain a comfortable finger spread (penalize weird poses).
   - Earn a bonus when the object reaches the goal threshold.

Extra rules:
- **Action penalty:** Squares of all actions to discourage flailing.
- **Hold flag:** Only when both palm and fingers are close do we switch from “init” to “grasp” scoring, exactly like the upstream code.
- **Logging:** We mirror the upstream metric names (e.g., `reward/init/right_hand_dist`, `reward/grasp/goal_rew`) so TensorBoard plots look the same.

Even if you’re a kid (or writing kid-friendly docs), remember: the observation slots, action layout, and reward pieces now line up with the released UniGraspTransformer task, so training scripts from the README behave the same way here.***

## Upstream Dedicated Policy Pipeline (Step 1 Recap)

Picture the original Microsoft setup as a school schedule:

1. **Pick a single object:** Choose one line from `train_set_results.yaml`. That line encodes which mesh (e.g., “mug / scale 0.08”) to load. When you run `python run_online.py ... --start_line N --end_line N+1`, the task only spawns that object across all environments.

2. **StateBasedGrasp task loads materials:** Inside `reference/UniGraspTransformer/dexgrasp/tasks/state_based_grasp.py`, the task:
   - Loads ShadowHand assets, the chosen object mesh, and its PCA axes/PointNet features.
   - Builds the same 400-D observation vector described above.
   - Hooks up reward shaping identical to what we mirrored.

3. **PPO training (`run_online.py`):**
   - `get_args()` parses CLI flags (task name, object file, env counts, etc.).
   - `load_cfg()` merges task YAML (`shadow_hand_grasp.yaml`) with training YAML (`dedicated_policy.yaml`) and sets log directories.
   - `parse_task()` instantiates `StateBasedGrasp`, pointing it at the selected object list.
   - `process_ppo()` builds the PPO runner (actor/critic networks sized per `Models` in the YAML).
   - PPO gathers rollouts across `num_envs=1000`, computes advantages, and updates weights for `max_iterations=10000`.

4. **Testing the new policy:** Immediately after training, the README runs a second `run_online.py ... --test --test_iteration 1` command. This reloads the fresh checkpoint and executes one evaluation pass to report success rate, without further training.

5. **Repeat per object:** Their helper script `run_online_parallel.sh` simply loops over line ranges (e.g., 0–9) and repeats steps 1–4 on different GPUs. Each object gets its own “dedicated” PPO policy saved under `Logs/Results/results_train/<object_id>`.

That’s the chunk we’re replicating now: a single-object PPO run using the `StateBasedGrasp` task. Later steps (trajectory saving and universal policy distillation) layer on top of those dedicated checkpoints.

## Observation Visualizer (Debug Tool)

Need to peek at the 400-D observation vector in real time? We added a GUI-based helper:

1. **Run a scene or test script without `--headless`** (you need a display).
2. **Set environment variables** before launching, e.g.
   ```bash
   UNIGRASP_OBS_TABLE=1 UNIGRASP_OBS_ENV=0 python legged_lab/scripts/unigrasptransformer/tests/test_spawn_scene.py
   ```
   - `UNIGRASP_OBS_TABLE=1` turns the visualizer on.
   - `UNIGRASP_OBS_ENV=<index>` chooses which environment row to show (default 0).
3. **Matplotlib windows pop out**, one per observation section. Each table lists the human-readable label (`palm_trans_x`, `goal_vec_z`, `obj_feat_42`, etc.) and the current value. Windows auto-refresh every sim step.

Need to inspect reward components too? Set `UNIGRASP_REWARD_TABLE=1` (and optionally `UNIGRASP_REWARD_ENV=<idx>`) alongside the command above and a “Reward Table” window will list every reward term (`reward/init/...`, `reward/grasp/...`, penalties) for the chosen environment.

Use this when calibrating reward weights, checking point-cloud inputs, or making sure the observation layout matches upstream. Turn it off (`UNIGRASP_OBS_TABLE=0`) before long PPO runs to avoid GUI overhead.
