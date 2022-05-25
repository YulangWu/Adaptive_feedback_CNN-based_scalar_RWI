# CNN-RWI
Implementation for CNN-RWI on TensorFlow 1.0 and PySIT platforms 

##  Prerequisites 
  1. MATLAB (https://www.mathworks.com/products/matlab.html) for prior model parcellation to create training models     
  2. PySIT (https://github.com/pysit/pysit) for reverse-time migration   
  3. TensorFlow r1.15 (https://www.tensorflow.org/) for CNN training and prediction    

##  User guide 
 1. Modify the parameters in the all_in.sh (shell script) 
 2. Modify the parameters in the RTM.py (provided under pysit1.0/) 
 3. Modify the parameters in the ML7_parse_vp.py (provided under CNN_approximation1.0/) 
 4. Run the shell script (all_in.sh)
 5. Run the MATLAB script (plot_itermediate_velocity.m) (provided under matlab1.0/Marmousi/) to plot inverted models
