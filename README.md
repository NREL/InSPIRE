<img src="docs/images_wiki/CE-MFC.png" width="400">

<table>
<tr>
  <td>License</td>
  <td>
    <a href="https://github.com/NREL/InSPIRE/blob/main/LICENSE.md">
    <img src="https://img.shields.io/pypi/l/pvlib.svg" alt="license" />
    </a>
</td>
</tr>
</table>


# InSPIRE Github Repository

Here we will collect scripts, trainings, studies, and any other open-source 
material that can help further the understanding and research of agrivoltaics.


Installation
=============

This repository does not have (at the moment) any functions, but it runs by
leveraging various open-source tools from NREL or others, such as 
bifacial_radiance, SAM, pvlib, etc. 

To install the necessary packages, we suggest you create a dedicated environment:

### Locally

You can also run the tutorial locally with
[miniconda](https://docs.conda.io/en/latest/miniconda.html) by following thes
steps:

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html).

1. Clone the repository:

   ```
   git clone https://github.com/NREL/InSPIRE.git
   ```

1. Create the environment and install the requirements. The repository includes
   a `requirements.txt` file that contains a list the packages needed to run
   the tutorials and some of the scripts (more below). 
   To install them using conda run:

   ```
   conda create -n inspire jupyter -c pvlib --file requirements.txt
   conda activate inspire
   ```

1. If you are interested on the tutorials or some of the studies that use Jupyter,
you will then start a Jupyter session:

   ```
   jupyter notebook
   ```

1. And Use the file explorer in Jupyter lab to browse to the tutorial or studies.


Please note that some of the tutorials that use bifacial_radiance requires the 
installation of the Radiance package and setup of various environment paths as 
detailed here:

  [bifacial_radiance installation](https://bifacial-radiance.readthedocs.io/en/latest/user_guide/installation.html)

InSPIRE is compatible with Python 3.9 and above.


Contributing
============

We need your help to make InSPIRE a great repository of AgriPV knowledge! 
Contact the github admins and teammembers to contribute. Since we cooperate 
with variuos undergraduate and graduate programs across the US, we do have a
[Copyright License Agreement](https://github.com/NREL/InSPIRE/blob/main/cla-1.0.md) and [instructions to sign it](https://github.com/NREL/InSPIRE/blob/main/sign-cla.md) that contributors might have to review with their
institutions.



License
=======

BSD 3-clause


Getting support
===============

If you suspect that you may have discovered a bug or if you'd like to
change something please make an issue on our
[GitHub issues page](https://github.com/NREL/InSPIRE/issues).


Citing
======

If you use the scripts included here in a publication, make sure you cite the 
individual tools used and their DOIs, as well as HPC use acknowledgment.