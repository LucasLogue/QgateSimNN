# uberpwnage
Qbit simulation
Setup-
    #1 Run the Setup: conda env create -f environment.yml
    #2 Then Activate: conda activate superballs-env (may need to conda init)
    #3 (Recommended for ML on NVIDIA Gpus) Install GPU Pytorch: conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
        Faster download: First ||| conda install -n base -c conda-forge mamba ||| then -> ||| mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

    For NVIDIA Brev
    We need WSL (installs ubuntu automatically), and to turn on virtual environments in bios
    Then run
    sudo bash -c "$(curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh)"
    in the linux terminal

    Login if needed
    brev login

    To link the instance to a vscode SSH
    brev open physproj

    Once in, check if micromamba is in installed
        If not
        curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
        export PATH=~/bin:$PATH
        alias mamba=micromamba
    Check if our balls-env is there
        If not
        mamba env create -f environment.yml

    To use micromamba
    eval "$(micromamba shell hook --shell bash)"

    If the stupid pytorch isn't working and not detecting GPU
    micromamba remove pytorch
    pip uninstall -y torch torchvision torchaudio
    pip install --no-cache-dir \
    torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu118
    
Scripts-
    driver.py----------------------------------------------------------------
    Currently at the top of the script we set parameters for the gate,
    electron configuration (and the attached parameters)
    Main unpacks parameters, sets up the initial values for the CMAES RL agent
    (inital_guess, std devs, bounds)
    It then runs the CMAES with the objective function script, and the parameters

    objective_function is the function that executes the goal of continously run through the gate operations and the solver from the parameters being tested by the RL agent. It calls the target gate, builds the electron configuration and so forth



Optimizations-
    in driver.py we need to stop reconstantly rebuilding the electron configuration
    In general, also need to be caching all nonvolative variables

    Physics Wise ------------
    Switch from gaussian wavepacket to Blackman
    Add dual frequency (fundamental and side band)
    Increase to vector for 5 to 10 waves at a time

    Calculation Wise--------
    Polish final gradient
    CMAES -> Skopt or Gkopt
    Check the control to response transfer with FFT
