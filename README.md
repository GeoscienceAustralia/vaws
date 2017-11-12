## Installation

It is recommended that you create a `conda` environment to run the *vaws* code.
These instructions have been tested on `Windows 7`, `Linux`, and `OS 10.11.x` and
is expected to work on most of modern operating systems.

1. Download and install [Miniconda](https://conda.io/miniconda.html) with Python 2.7.

 * Windows
    - Double-click the downloaded `Miniconda2-latest-Windows-x86_64.exe` file.
	- When installation is finished, from the `Start` menu, open the `Anaconda Prompt`.

 * Linux/Mac
    - In your Terminal window, run:   
    
        `bash Miniconda2-latest-Linux-x86_64.sh` for `Linux` or 
        
        `bash Miniconda2-latest-MacOSX-x86_64.sh` for `OS X`

2. Create a conda environment. 

    In the terminal client, enter the following to create the environment called *vaws_env*.

    `conda create -n vaws_env python=2.7`

3. Activate the environment.

    In the terminal client, enter the following to activate the environment.

    * Windows
 
        `activate vaws_env`

     * Linux/Mac
 
        `source activate vaws_env`

4. Install the *vaws* code from conda channel

    In the terminal client, enter the following to install the *vaws*.

    `conda install -c crankymax vaws`

    In case you see `PackageNotFoundError: Packages missing in current channels:` then enter the following and try above command again.

    `conda config --add channels conda-forge`

5. Run the vaws code

    In the terminal client, enter the following to run the code.

    `vaws`

## Running the tests

To run tests use `unittest`:
    
    cd vaws
    python -m unittest discover model/tests/

