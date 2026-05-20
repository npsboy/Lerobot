@echo off

:loop

if exist "C:\Users\Tusha\OneDrive\Desktop\mycode\WIP\Lerobot\eval_act_test" (
    rmdir /s /q "C:\Users\Tusha\OneDrive\Desktop\mycode\WIP\Lerobot\eval_act_test"
)

lerobot-record ^
--robot.type=so101_follower ^
--robot.id=my_follower_arm ^
--robot.port=COM6 ^
--robot.calibration_dir="C:\Users\Tusha\.cache\huggingface\lerobot\calibration\robots\so_follower" ^
--robot.cameras "{\"front\":{\"type\":\"opencv\",\"index_or_path\":1,\"width\":640,\"height\":480,\"fps\":30},\"side\":{\"type\":\"opencv\",\"index_or_path\":2,\"width\":640,\"height\":480,\"fps\":30}}" ^
--display_data=false ^
--dataset.repo_id=local/eval_act_test ^
--dataset.root="C:\Users\Tusha\OneDrive\Desktop\mycode\WIP\Lerobot\eval_act_test" ^
--dataset.single_task="pick up object" ^
--dataset.push_to_hub=false ^
--dataset.episode_time_s=20 ^
--dataset.reset_time_s=5 ^
--dataset.num_episodes=1 ^
--policy.type=act ^
--policy.device=cpu ^
--policy.pretrained_path="C:\Users\Tusha\OneDrive\Desktop\mycode\WIP\Lerobot\outputs\train\double_cam_test\checkpoints\020000\pretrained_model"

echo.
pause

goto loop