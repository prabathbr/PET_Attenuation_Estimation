## 📂 Models

- `Our_Model/`  
  Final proposed model that directly estimates attenuation information from sinograms, avoiding intermediate PET reconstruction.

- `Baseline/`  
  Baseline model using non-attenuation-corrected PET images as input for comparison.

- `Ablation_Direct_Skip/`  
  Ablation model with direct skip connections.

- `Ablation_No_Skip/`  
  Ablation model without skip connections.

## 📜 Scripts in Each Folder

Every model folder contains the following files:

- `Train.py`  
  Trains the model using the training dataset.

- `Test.py`  
  Generates predictions on the holdout test dataset.

- `Test_Evaluation.py`  
  Computes quantitative evaluation metrics for the test set predictions.
