# acBF-aberration-corrected-bright-field-STEM
The repository hosts the code for aberration-corrected bright-field (acBF-STEM) imaging and python scripts to reproduce figures in manuscripts [1][2].

Start with acBF_walkthrough.ipynb, which reproduces Figure 5 of [2] (data source: https://doi.org/10.5281/zenodo.15283331).

This acBF code uses py4DSTEM as a preparatory step to estiamte the shifts and the aberration function.

CIF file(s) to reproduce Figure 4 and 7 are included in folder "CIFs". 

[1] Ma, D., Li, G., Muller, D. A., & Zeltmann, S. E. (2025). Information in 4D-STEM: Where it is, and How to Use it. arXiv preprint arXiv:2507.21034.
[2] Ma, D., Muller, D. A., & Zeltmann, S. E. (2025). Using Aberrations to Improve Dose-Efficient Tilt-corrected 4D-STEM Imaging. arXiv preprint arXiv:2510.01493.


## Author 
Desheng Ma (dm852@cornell.edu)

Steven E Zeltmann (steven.zeltmann@cornell.edu)

Developed at the Muller Group, Cornell University.

### Dependencies:
- py4DSTEM 0.14.8
- abTEM-legacy 1.0.0.b34
## Acknowledgement 
- [tcBF-STEM](https://github.com/yyu2017/tcBFSTEM)  
- [py4DSTEM](https://github.com/py4dstem/py4DSTEM)
