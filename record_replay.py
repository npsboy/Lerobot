import pprint
import lerobot

print("lerobot dir = ", dir(lerobot))

# robot = make_robot("your_robot_config")  # both leader + follower setup
# 
# dataset = LeRobotDataset.create(
#     repo_id="local/test_recording",
# )
# 
# robot.connect()
# 
# print("Recording... move the leader arm")
# 
# for i in range(200):  # ~10 seconds at 20 Hz
#     obs = robot.get_observation()
# 
#     # this reads leader + applies to follower internally
#     action = robot.teleop_step()
# 
#     dataset.add_frame(
#         observation=obs,
#         action=action,
#     )
# 
# dataset.save()
# robot.disconnect()
