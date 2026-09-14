# Behavior trees

Ready-to-import Robo Boy Behavior Trees for the Microduck simulation. They live here,
next to `../ros/microduck_interfaces`, so `tests/test_behavior_trees.py` can check every
action node against the interface it actually binds to: an action name the bridge
serves, a matching action type, and goal fields that exist in the `.action` file.

Import one from Robo Boy's Behavior Tree toolbar, then press play with the simulator
running. Robo Boy discovers the goal fields through `/rosapi/action_goal_details`, so
the `microduck-interfaces` overlay must be mounted into its rosbridge container.

| Tree | Demonstrates |
| --- | --- |
| `microduck-move-forward.json` | A single `MoveRelative` goal. The node stays Running for the whole walk and only succeeds once the simulator reports arrival. |
| `microduck-forward-turn-forward.json` | Three goals in a Sequence. Each has to finish before the next starts. |
| `microduck-square-patrol.json` | A Repeat driving one side and one corner four times. Errors do not compound because each goal reports its real completion. |
| `microduck-policy-then-move.json` | A trained control policy triggered as an action, waited on to its own completion condition, then a relative move. |
| `microduck-failure-recovery.json` | An unreachable move under a Timeout. The timeout cancels the goal over rosbridge, the simulator reports it canceled, and a Selector falls back to a safe posture. |

`microduck-move-forward` is the quickest check that the action layer is wired up
correctly end to end. If it turns green the moment you press play instead of after the
robot walks, the stale `microduck_interfaces` overlay is still mounted.

Two further trees covering policy switching and the mouth service ship with the
optional control panel in
[`../../robo-boy-microduck-control-panel/behavior-trees`](../../robo-boy-microduck-control-panel/behavior-trees).
