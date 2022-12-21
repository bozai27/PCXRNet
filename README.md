# PCXRNet

This repository contains the PyTorch implementation of our network proposed in the paper "PCXRNet: Condense attention module and Multiconvolution spatial attention module for Pneumonia Chest X-Ray detection"


## Requirements
- python		3.6.13
- numpy	   	  1.19.5		
- pytorch	  	1.5.1
- torchvision	 0.6.1
- opencv-python	4.5.1


## Usage
1. Clone the repository

2. Download the dataset COVID19-v4 from the [link](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) to the  `data` directory

3. you can  test the model directly by the following commands

   ```
   ./test_COVID-19.sh
   ```

   


## Citation
Y. Feng, X. Yang, D. Qiu, H. Zhang, D. Wei and J. Liu, "PCXRNet: Pneumonia Diagnosis From Chest X-Ray Images Using Condense Attention Block and Multiconvolution Attention Block," in IEEE Journal of Biomedical and Health Informatics, vol. 26, no. 4, pp. 1484-1495, April 2022, doi: 10.1109/JBHI.2022.3148317.
