# uberpwnage
Qbit simulation
Setup-
    #1 Run the Setup: conda env create -f environment.yml
    #2 Then Activate: conda activate superballs-env (may need to conda init)
    #3 (Recommended for ML on NVIDIA Gpus) Install GPU Pytorch: conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
        Faster download: First ||| conda install -n base -c conda-forge mamba ||| then -> ||| mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    
Scripts-
    tdse_pinn_test.py:
        I only moderately used my brain on this, AI-gooned most of the hard stuff.
    tdse_adaptive.py:
        This is just gemini's code, I don't advise running it until I refactor, it blew up my laptop earlier today
