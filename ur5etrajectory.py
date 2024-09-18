import mujoco
import mujoco.viewer
import numpy as np
import time


x = np.load('cup.npy', allow_pickle=True)
robot_state_vectors = x[0]['robot_state']
    

dt: float = 0.002

def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    
    model = mujoco.MjModel.from_xml_path("universal_robots_ur5e/scene.xml")
    data = mujoco.MjData(model)

    
    model.opt.timestep = dt

    
    robot_state_index = 0
    num_states = len(robot_state_vectors)

    
    num_joints = 6  

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        
        key_id = model.key("home").id
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()

            
            data.qpos[:num_joints] = robot_state_vectors[robot_state_index, :num_joints]

            
            mujoco.mj_forward(model, data)

            
            robot_state_index += 1
            if robot_state_index >= num_states:
                robot_state_index = 0  

            viewer.sync()

            
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(0.01)

if __name__ == "__main__":
    main()
