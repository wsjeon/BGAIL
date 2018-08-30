## Environment setting
We set our environment as follows:
1.  We already include OpenAI's gym, baselines, mujoco_py in this repository.
    - All cloned from OpenAI's github on Aug. 24th, 2018.
2.  Install Anaconda3 and create `bgail` environment.
    ```bash
    conda create -n bgail python=3.6
    source activate bgail
    ```
3.  Install MuJoCo.
    - Obtain a 30-day free trial on the MuJoCo website 
    or free license if you are a student.
    The license key will arrive in an email with your username
    and password.
    - Download the MuJoCo version 1.50 binaries for Linux, OSX, or Windows.
    - Unzip the downloaded mjpro150 directory into `$HOME/.mujoco/mjpro150`, 
    and place your license key (the `mjkey.txt` file from your email) at `$HOME/.mujoco/mjkey.txt`.
    - Install `mujoco_py`.
    ```bash
    cd dependencies/mujoco_py/
    LD_LIBRARY_PATH=$HOME/.mujoco/mjpro150/bin pip install -U 'mujoco-py<1.50.2,>=1.50.1'
    ```
    - Do the following steps and follow the message.
    ```
    $ python
    import mujoco_py
    from os.path import dirname
    model = mujoco_py.load_model_from_path(dirname(dirname(mujoco_py.__file__))  +"/xmls/claw.xml")
    sim = mujoco_py.MjSim(model)
    
    print(sim.data.qpos)
    # [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    
    sim.step()
    print(sim.data.qpos)
    # [  2.09217903e-06  -1.82329050e-12  -1.16711384e-07  -4.69613872e-11
    #   -1.43931860e-05   4.73350204e-10  -3.23749942e-05  -1.19854057e-13
    #   -2.39251380e-08  -4.46750545e-07   1.78771599e-09  -1.04232280e-08]
    ```

4. Install Gym.
    ```bash
    cd ../gym/
    pip install -e .[classic_control,mujoco]
    ```

5. Install TensorFlow 1.10.0 with GPU support.
    ```bash
    pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.0-cp36-cp36m-linux_x86_64.whl
    ```

6. Install Baselines.
    ```bash
    cd ../baselines/
    pip install -e .
    ```
    
7. Test
    ```bash
    python -m baselines.run --alg=trpo_mpi --env=Hopper-v2
    ```
    -   Rendering problem: 
        When we add `env.render()` in *line 60* of `baselines/trpo_mpi/trpo_mpi.py`,
        we found the following error:
        ```
        ERROR: GLEW initalization error: Missing GL version
        ```
        To solve this problem, we use the comment [here](https://github.com/openai/mujoco-py/issues/44#issuecomment-399679237).

8. Install `h5py`.
    ```bash
    conda install h5py
    ```        
