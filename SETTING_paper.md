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
    - Download the MuJoCo version 1.31 binaries for Linux, OSX, or Windows.
    - Unzip the downloaded mjpro131 directory into `$HOME/.mujoco/mjpro131`, 
    and place your license key (the `mjkey.txt` file from your email) at `$HOME/.mujoco/mjkey.txt`.
    - Install `mujoco-py.
    ```bash
    pip install mujoco-py==0.5.7
    ```

4. Install Gym.
    ```bash
    pip install gym==0.9.0 
    ```

5. Install TensorFlow 1.10.0 with GPU support.
    ```bash
    pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.0-cp36-cp36m-linux_x86_64.whl
    ```

6. Install `baselines 0.1.5` (with little modification)
    ```bash
    cd dependencies/baselines/
    pip install -e .
    ```
    
7. Install `h5py`.
    ```bash
    conda install h5py
    ```        
