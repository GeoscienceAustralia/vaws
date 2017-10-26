#### Conda setup

It is recommended that you create a virtual conda environment to run the `vaws` code.
These instructions have been tested on `Redhat EL 6.7` and is expected to work for most newer versions of `Linux`. 
Conda and the requirements can be installed using the following steps.

1. Download and install [Miniconda](https://conda.io/miniconda.html) with Python 2.7.

* Windows
- Double-click the downloaded .exe file.
- Follow the instructions on the screen.
- If you are unsure about any setting, accept the defaults. You can change them later.
- When installation is finished, from the Start menu, open the Anaconda Prompt.

* Linux/Mac
- In your Terminal window, run:
>```bash Miniconda3-latest-MacOSX-x86_64.sh```
>```bash Miniconda3-latest-Linux-x86_64.sh```

2. Once Miniconda is installed, you can use the conda command to create an environment called vaws_env.

* Windows
Click 'Anaconda Prompt' in the Windows Start

>``` ~/miniconda2/bin/conda create -n vaws_env python=2.7
>``` conda env create --name vaws_env --file vaws.yml```

> *Note: for windows use conda env create --name vaws_env --file vaws_win.yml*

4. Activate the environment with

>``` source activate vaws env```

> *Note: for windows use activate vaws_env*

Then use the build script for your environment 
* Linux/Mac

    cd src/gui  
    ./build.sh
     
* Windows

    cd src\gui  
    build.cmd

#### Install from conda channel

~/miniconda2/bin/conda create -n vaws_env python=2.7 
source ~/miniconda2/bin/activate vaws_env
conda install -c crankymax vaws

#### Inputs required to run the vaws:
These files are not in repo and you will need access to these files to be able to run the code.

* Shapefiles: Hyeuk to describe.
* glenda_reduced: Hyeuk to describe.
* input: Hyeuk to describe.


#### How to run the vaws code

Running the vaws code is simple.
    
    source ~/miniconda2/bin/activate vaws_env
    vaws

#### Run tests
To run tests use either `nose` or `unittest`:
    
    cd vaws
    python -m unittest discover transmission/tests/
    or
    nosetests

#### Parallel vs Serial run
A dedicated config file has not been implemented yet and the configuration is managed by the `TransmissionConfig` class inside the `config_class.py`. The value `self.parallel = 1` indicates that Monte Carlo simulations will be performed in parallel using all the (hyperthreaded) cores available on the computer. To change to serial computation, simply change to `self.parallel = 0` instead.

#### Building the VAWS conda package

Prerequisites:

    conda
    conda-build
    anaconda-client

Build the package

    cd <vaws dir>/build
    ~/miniconda2/bin/conda-build .

Upload to conda-forge

    anaconda login
    anaconda upload /home/ubuntu/miniconda2/conda-bld/linux-64/vaws-2.0-py27<package details>

Windows upload to conda

    \dev\Miniconda2\Scripts\anaconda -s anaconda.org login
    \dev\Miniconda2\Scripts\anaconda upload --force c:\dev\Miniconda2\conda-bld\win-64\vaws-2.0-py27<package details>
