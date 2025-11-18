# UniGraspTransformer RL Notes (Kid-Friendly Edition)

Imagine the robot hand is a kid stacking blocks. It needs to **see**, **move**, and **score points**. Below is how both our Isaac Lab port and the original Microsoft Research task teach the kid.

## Observation Space (What the hand "sees")

- **Big picture:** Every time step we build a 400-number list. Think of it as a long sticker sheet with labeled sections.
- **Sections we share with the upstream task:**
  - **Hand joints (0–166):** 22 finger joints × (position, speed, pretend force). We center positions around the default pose just like upstream.
  - **Fingertips (66–161):** For each tip we store where it is, which way it points, and how it moves. We rotate everything into the object’s frame so the hand always thinks “object at center,” mirroring `StateBasedGrasp`.
  - **Palm pose (161–167):** Palm position + yaw/pitch/roll inside the object frame.
  - **Last action (167–191):** Previous palm motion + finger deltas, also expressed relative to the object.
  - **Object block (191–207+):** In the reference code this carries velocities, goal offsets, and optional PCA axes centered in the object frame. We match that layout: pose slots are zero (because we are already in object coordinates), velocities and goal vector are rotated into the same frame.
  - **Object visual features (207–335):** Upstream fills this with PointNet embeddings (static or dynamic depending on config). Our port drops in centered point-cloud samples when we have them, or zeros when we don’t. Hooking the PN encoder/scaler would make this 1:1.
  - **Time code (335–364):** Original task uses `progress_buf` plus a sinusoidal encoding. We now copy that exact formula so every time step gets the same “rhythm” stickers.
  - **Hand vs. object distances (364–400):** Upstream computes min distances from many hand bodies to the object cloud. We reproduce that when we have the cloud, otherwise we write zeros so PPO still knows the slot exists.

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
