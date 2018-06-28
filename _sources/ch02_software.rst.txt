***************
Getting Started
***************

..
  # with overline, for parts
  * with overline, for chapters
  =, for sections
  -, for subsections
  ^, for subsubsections
  ", for paragraphs

This chapter provides instructions on how to install and run the code for general users. Also it provides instructions for developers on how to install, test and build the package of the code. These instructions have been tested on `Windows 7`, `Linux`, and `OS 10.11.x` and is expected to work on most of modern operating systems.

Instructions for general users
==============================

Installation
------------

The VAWS code currently runs with Python 2.7 with many dependencies. It is recommended to create a Python environment dedicated to the code without disrupting the existing environment. With conda, you can manage environments easily. Instructions below are based on conda, but virutalenv can be used alternatively.

.. _install_conda:

1. Install Miniconda

Download and install Miniconda(https://conda.io/miniconda.html) with Python 2.7. This step can be skipped if either Miniconda or Anaconda with Python 2.7 is already installed.

  * Windows

    - Double-click the downloaded `Miniconda2-latest-Windows-x86_64.exe` file.
    - When installation is finished, from the `Start` menu, open the `Anaconda Prompt`.

  * Linux

    - In Terminal window, run

      .. code-block:: bash

        $ bash Miniconda2-latest-Linux-x86_64.sh

  * Mac

    - In Terminal window, run

      .. code-block:: bash

        $ bash Miniconda2-latest-MacOSX-x86_64.sh


2. Create a conda environment.

In the terminal client, enter the following command to create the environment called *vaws_env*.

  .. code-block:: bash

    conda create -n vaws_env python=2.7

.. _activate_environment:

3. Activate the environment.

In the terminal client, enter the following to activate the environment.

  * Windows

    .. code-block:: bash

      activate vaws_env

  * Linux/Mac

    .. code-block:: bash

      source activate vaws_env

.. _install_code:

4. Install the code from conda channel

In the terminal client, enter the following to install the code.

  .. code-block:: bash

    conda install -c dynaryu vaws

In case you see `PackageNotFoundError: Packages missing in current channels:` then enter the following in the terminal client and try above command again.

  .. code-block:: bash

    conda config --add channels conda-forge

Updating
--------

In case new version of the code is available, you may update the code. The conda environment *vaws_env* should be activated first as :ref:`2.1.1 step 3 <activate_environment>`. And then enter the following commands in the terminal to remove the old version and re-install the new version of the code.

  .. code-block:: bash

    conda remove vaws
    conda install -c dynaryu vaws


Running through GUI
-------------------

.. _run_code:

To run the code, the conda environment *vaws_env* should be activated first as :ref:`2.1.1 step 3 <activate_environment>`. And then enter the following command in the terminal.

  .. code-block:: bash

    vaws

The default scenario will be loaded as shown in :numref:`main_window_fig`. See :ref:`chapter 4 <use_of_the_GUI>` for details.

.. _main_window_fig:
.. figure:: _static/image/main_window.png
    :align: center
    :width: 80 %

    Program main window with default scenario loaded


Instructions for developers
===========================

The development of the code is tracked using the git version control system. The source code is at `https://github.com/GeoscienceAustralia/vaws <https://github.com/GeoscienceAustralia/vaws>`_.

Installation
------------

1. Get the source code

Source code can be copied by cloning the git repository or downloading the zip file from the git repository.

  * If git is installed, run the following command in the terminal

    .. code-block:: bash

      $ git clone git@github.com:GeoscienceAustralia/vaws.git

  * Otherwise download the zip file (https://github.com/GeoscienceAustralia/vaws/archive/master.zip) and then extract it.

This step will create directory called <vaws dir>.

2. Create a conda environment.

Make sure either miniconda or anaconda is installed. Otherwise install either Miniconda or Anaconda with Python 2.7 as :ref:`2.1.1 step 1 <install_conda>`. Then create the environment called *vaws_env*. by entering the following command in the terminal.

  * Windows

    .. code-block:: bash

      cd <vaws dir>
      conda env create --name vaws_env --file vaws_win.yml

  * Linux/Mac

    .. code-block:: bash

      cd <vaws dir>
      conda env create --name vaws_env --file vaws_env.yml

This will create the environment called vaws_env. The *vaws_env* can be activated as :ref:`2.1.1 step 3 <activate_environment>`.

3. Create GUI

To create the GUI of the code, enter the following commands in the terminal.

  * Windows

    ::

      cd <vaws dir>\vaws\gui
      build.cmd

  * Linux/Mac

    .. code-block:: bash

      cd <vaws dir>/vaws/gui
      ./build.sh

4. Run the code

The code can be run in either GUI or CLI mode.

  * GUI

    ::

      cd <vaws dir>
      python -m vaws.gui.main  # for default scenario
      python -m vaws.gui.main -c <config_file>  # for a specific scenario

  * CLI

    ::

      cd <vaws dir>
      python -m vaws.model.main -c <config_file>  # for a specific scenario

Building the conda package
--------------------------

Steps for the conda package is described below. Please refer to (https://conda.io/docs/user-guide/tutorials/build-pkgs.html) for details.

1. Install conda-build and anaconda-client

To build the package, you need to install `conda-build` and `anaaconda-client` in the conda *root* environment not the `vaws_env` environment. And then enter the following in the terminal.

  .. code-block:: bash

    conda install conda-build anaconda-client

2. Build the package

In the terminal client, enter the following to build the package.

  .. code-block:: bash

    cd <vaws dir>/build
    conda-build .

At the end of the building, you should see something like below:

  .. code-block:: bash

    Updating index at /foo/anaconda2/conda-bld/noarch to make package installable with dependencies
    INFO:conda_build.build:Updating index at /foo/anaconda2/conda-bld/noarch to make package installable with dependencies
    Nothing to test for: /foo/anaconda2/conda-bld/osx-64/vaws-2.0.3-py27_1.tar.bz2
    # Automatic uploading is disabled
    # If you want to upload package(s) to anaconda.org later, type:

    anaconda upload /foo/anaconda2/conda-bld/osx-64/vaws-2.0.3-py27_1.tar.bz2

    # To have conda build upload to anaconda.org automatically, use
    # $ conda config --set anaconda_upload yes

    anaconda_upload is not set.  Not uploading wheels: []


3. Upload to anaconda channel

In the terminal client, enter the following to upload the package to the channel.

  .. code-block:: bash

    anaconda login
    anaconda upload <package>

Testing the code
----------------

To test the code, the conda environment *vaws_env* should be activated first as :ref:`2.1.1 step 3 <activate_environment>`. And then enter the following command in the terminal.

.. code-block:: bash

  nosetests -v vaws

You should see something similar to below.

.. code-block:: bash

  test_distribute_damage_by_row (vaws.model.tests.test_simulation_batten.TestHouseDamage) ... ok
  test_calc (vaws.model.tests.test_stats.MyTestCase) ... ok
  test_calc2 (vaws.model.tests.test_stats.MyTestCase) ... ok
  test_calc_big_a_b_values (vaws.model.tests.test_stats.MyTestCase) ... ok
  test_compute_arithmetic_mean_stdev (vaws.model.tests.test_stats.MyTestCase) ... ok
  test_compute_logarithmic_mean_stdev (vaws.model.tests.test_stats.MyTestCase) ... ok
  test_gev_calc (vaws.model.tests.test_stats.MyTestCase) ... ok
  test_gev_calc2 (vaws.model.tests.test_stats.MyTestCase) ... ok
  test_sample_logrnormal (vaws.model.tests.test_stats.MyTestCase) ... ok
  test_calc_zone_pressures (vaws.model.tests.test_zone.MyTestCase) ... ok
  test_get_grid (vaws.model.tests.test_zone.MyTestCase) ... ok
  test_is_wall (vaws.model.tests.test_zone.MyTestCase) ... ok
  test_str2num (vaws.model.tests.test_zone.MyTestCase) ... ok

  ----------------------------------------------------------------------
  Ran 93 tests in 56.053s

  OK

Documentation
-------------

The VAWS user manual is generated using the Sphinx documentation tool (http://www.sphinx-doc.org/en/1.7/). The source code of the manual is located at the subdirectory called *docs* under *vaws* root directory. Both the *html* and *pdf* versions of the manual can be generated using the commands below:

.. code-block:: bash

    cd <vaws dir>/docs
    make html   # for html format
    make latexpdf  # for pdf format

The html version of the manual can be updated to the manual website (`http://geoscienceaustralia.github.io/vaws`_) as below. Note that the <build_dir> can be set in the *Makefile* in the *docs* directory.

.. code-block:: bash

    cd <build_dir>/html
    git checkout gh-pages
    git commit -m <commit_message>
    git push origin gh-pages

The pdf version of the manaual can be accessed at `https://github.com/GeoscienceAustralia/vaws/tree/master/manaual.pdf`_.

.. _https://github.com/GeoscienceAustralia/vaws/tree/master/manaual.pdf: https://github.com/GeoscienceAustralia/vaws/tree/master/manaual.pdf

.. _http://geoscienceaustralia.github.io/vaws: http://geoscienceaustralia.github.io/vaws

