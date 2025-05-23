Welcome to Assignment 2!

We'll be using PyTorch for this assignment. If you're not familiar with PyTorch, or if you would like to review some of the fundamentals of PyTorch, the PyTorch review session is posted on Canvas under Course Videos.  

If you want to continue using your cs224n environment from assignment 1 for this assignment, please make sure you have all the dependencies listed in local_env.yml. To do so, please run: 

# 1. Activate your old environment:

    conda activate cs224n

# 2. Install docopt

    conda install docopt

# 3. Install pytorch, torchvision, and tqdm

    conda install pytorch torchvision -c pytorch
    conda install -c anaconda tqdm


If you would like to instead create a new environment for this assignment, please run:

# 1. Create an environment with dependencies specified in local_env.yml (note that this can take some time depending on your laptop):
    
    conda env create -f local_env.yml

# 2. Activate the new environment:
    
    conda activate cs224n_a2
    

# To deactivate an active environment, use
    
    conda deactivate

## No `zip` command for Windows
If you are using Windows, you may see an error when running the `collect_submission.sh` script. This is because the `zip` command is not available by default on Windows. If you encounter this error, you can try other methods mentioned [here](https://superuser.com/questions/201371/create-zip-folder-from-the-command-line-windows). Alternatively, you can manually zip the Python files.