@echo off

rmdir /s /q "%USERPROFILE%\.cache\huggingface\lerobot\npsboy\test_dataset2" 2>nul

lerobot-record ^
--robot.type=so101_follower ^
--robot.id=my_follower_arm ^
--robot.port=COM6 ^
--robot.cameras "{\"front\":{\"type\":\"opencv\",\"index_or_path\":1,\"width\":640,\"height\":480,\"fps\":15},\"wrist\":{\"type\":\"opencv\",\"index_or_path\":2,\"width\":640,\"height\":480,\"fps\":15}}" ^
--teleop.type=so101_leader ^
--teleop.id=my_leader_arm ^
--teleop.port=COM7 ^
--dataset.repo_id=npsboy/test_dataset2 ^
--dataset.single_task "pick up object" ^
--dataset.push_to_hub false ^
--display_data true ^
--dataset.episode_time_s=20 ^
--dataset.reset_time_s=2 ^
--dataset.num_episodes=70

pause