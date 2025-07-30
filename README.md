# uberpwnage
Qbit simulation
Setup-
    #1 Run the Setup: conda env create -f environment.yml
    #2 Then Activate: conda activate superballs-env (may need to conda init)
    #3 (Recommended for ML on NVIDIA Gpus) Install GPU Pytorch: conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
        Faster download: First ||| conda install -n base -c conda-forge mamba ||| then -> ||| mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    
Scripts-
    tdse_pinn_test.py:
        My absolute product of AI gooning! TBH used a limited amount of my own brain.
        Utilizies a PINN (Physics-Informed Neural Network): Takes 3 coordinates (x, y, t), using tanh operations outputs a 2 coordinate wavefunction (real, imaginary)
