## 🧪 Simulation

## 📂 Contents

### `Sample_Simulation_Config.yaml`

- A sample template configuration file for the [SimPET](https://github.com/txusser/simpet) simulation platform.
- This configuration generates both:
  - 3D PET sinogram
  - Non-attenuation-corrected PET images

> ℹ️ For installation instructions and parameter explanations, refer to the official SimPET repository:  
> https://github.com/txusser/simpet

---

### `ssrb_crawler.py`

- A utility script that performs batchwise Single Slice Rebinning (SSRB) on 3D sinograms to convert them into 2D PET sinograms.
- These 2D PET sinograms would be the input for training pipelines in the `Models/` folder.

#### Requirements
- STIR version 5.1 or higher 

> ℹ️ Installation instructions for STIR via Conda which contains SSRB binary:  
> https://github.com/UCL/STIR/wiki/Installing-STIR-with-conda

---

## 📦 Dataset Sources

The attenuation and activity map templates for SimPET simulations were obtained from the SimPET public datasets, retrieved on May 30, 2024, from:

- https://sourceforge.net/projects/simpet/

## 📚 Dataset Citations

1. **López-González et al. (2020)**  
   *Intensity normalization methods in brain FDG-PET quantification*  
   [NeuroImage, Vol 222](https://doi.org/10.1016/j.neuroimage.2020.117229)

   > López-González, F. J., Silva-Rodríguez, J., Paredes-Pacheco, J., et al. (2020).  
   > *NeuroImage*, 222, 117229.

2. **Paredes-Pacheco et al. (2021)**  
   *SimPET — An open online platform for the Monte Carlo simulation of realistic brain PET data*  
   [Medical Physics, Vol 48](https://doi.org/10.1002/mp.14838)

   > Paredes-Pacheco, J., López-González, F. J., Silva-Rodríguez, J., et al. (2021).  
   > *Medical Physics*, 48(5), 2482–2493.

