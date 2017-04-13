#### Conda setup setup
It is recommended that you create a virtual conda environment to run the `vaws` code.
These instructions have been tested on `Redhat EL 6.7` and is expected to work for most newer versions of `Linux`. 
Conda and the requirements can be installed using the following steps.

set HTTPS_PROXY=https://u53337:password@10.7.64.209:8080

1. Clone the VAWS repository 
>```git clone git@github.com:GeoscienceAustralia/vaws.git```
2. Download and install [Miniconda](https://conda.io/miniconda.html)
3. Create a virtual environment called vaws_env with  
>``` conda create --name vaws_env --file vaws.yml```

> *Note: for windows use conda create --name vaws_env --file vaws_win.yml*

4. Activate the environment with

>``` source activate vaws env```

> *Note: for windows use activate vaws_env*

Then use the build script for your environment 
* Linux/Mac

    cd gui  
    ./build.sh
     
* Windows

    cd gui  
    ./build.cmd

#### Inputs required to run the vaws:
These files are not in repo and you will need access to these files to be able to run the code.

* Shapefiles: Hyeuk to describe.
* glenda_reduced: Hyeuk to describe.
* input: Hyeuk to describe.


#### How to run the vaws code

Running the vaws code is simple.
    
    cd vaws
    ./vaws.sh
    or for windows
    ./vaws.cmd

#### Run tests
To run tests use either `nose` or `unittest`:
    
    cd vaws
    python -m unittest discover transmission/tests/
    or
    nosetests

#### Parallel vs Serial run
A dedicated config file has not been implemented yet and the configuration is managed by the `TransmissionConfig` class inside the `config_class.py`. The value `self.parallel = 1` indicates that Monte Carlo simulations will be performed in parallel using all the (hyperthreaded) cores available on the computer. To change to serial computation, simply change to `self.parallel = 0` instead.