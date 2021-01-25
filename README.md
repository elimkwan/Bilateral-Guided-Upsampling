The project showcased a python implementation of Bilateral Guided Upsampling introduced by J. Chen et al. (2016), and extended its application to Tone Mapping and Gradient Enhancement Operators. More explainations can be found from the report.pdf

## Running the code
The full pipeline is demonstrated in `main.ipynb`. A short demo is in `demo.ipynb` which uses the `main.py` as a module. 

## Experiment Results
High-resolution output from the experiments are documented in `./report`

## Main Dependencies
Basics
```
opencv
scikit-image
scipy
matplotlib
```
For solving least square problem
```
sudo apt-get install python-scipy libsuitesparse-dev
conda install -c conda-forge scikit-sparse
```
For Tone Mapping operators and TMQI
```
#pip install git+https://github.com/dvolgyes/TMQI
import imageio
imageio.plugins.freeimage.download()
from TMQI import TMQI
```
If some libraries are still missing, feel free to check out the `environment/environment_graphics.yml`