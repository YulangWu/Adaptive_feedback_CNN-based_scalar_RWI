#! /bin/sh
#======================================================================================
#      Adaptive feedback convolutional-neural-network-based  
#         high-resolution reflection-waveform inversion
#======================================================================================
# There are two parts in CNN-RWI: 
#
# Part 1 is responsible for creating prior information as CNN input data for
# the CNN prediction (the original starting model and RTm image from true model)
#
# Part 2 is an outer loop responsible for iteratively predict the velocity model 
# from the prior information and adaptively create dataset for CNN training
#======================================================================================

#======================================================================================
# Experiment setup 
num_threads=4  # Number of threads to obtain training RTM images in parallel
iteration=0    # The starting iteration of the CNN-RFWI
iter=50        # The total iteration number of the CNN-RFWI
num_samples=16 # The number of training samples used for CNN training
matlab_dir='matlab1.0/' # The directory to create training models by parcellation
vel_dir='Marmousi/'     # The directory to store the predicted models
pysit_dir='pysit1.0/'   # The directory to obtain RTM images on PySIT platform
CNN_dir='CNN_approximation1.0/' # The directory for CNN training and prediction
CNN_result_dir="CNN_results/"   # The directory to store intermediate results
CNN_max_epochs=1600             # Number of iterations for CNN training
store_weights_freq=1600         # The increment to store CNN weights
CNN_last_epochs=`expr $CNN_max_epochs - 1`
sh_num_smooth_iteration=400     # Number of iterations to smooth velocity model
sh_filter_size=3                # The size of the Gaussian filter
sh_water_depth=13               # The water depth (in grid)
nz=256                          # The number of grids in depth
nx=256                          # The number of grids in horizontal direction
ratio=0.01                      # The theshold of parcellation (percentage)
velocity_filename='Marmousi_vp256x256x10000z230.dat' # The filename of true model
sleep_time=60                   # For the use of synchronization 
#======================================================================================

#======================================================================================
# PART 1: Obtain the original starting velocity model and the corresponding RTM image
#         from the known true velocity model as prior information for CNN-RWI
#======================================================================================
# 1.1 Obtain the starting velocity model by smoothing the true velocity model
echo "1.1 Obtain the starting velocity model by smoothing the true velocity model"
cd $matlab_dir$vel_dir
  matlab_filename=${iteration}"th_mig_"
  matlab -nodesktop -nosplash -noFigureWindows -r \
  "sh_num_smooth_iteration=$sh_num_smooth_iteration,\
  sh_filename='$matlab_dir$vel_dir$velocity_filename',\
  sh_filter_size=$sh_filter_size,\
  sh_nx=$nx,\
  sh_nz=$nz,\
  sh_water_depth=$sh_water_depth;\
  CNN_RWI_starting_model;quit"
cd ../../

# 1.2 Prepare the true and starting velocity models for RTM
echo "1.2 Prepare the true and starting velocity models for RTM"
cd matlab1.0
  matlab_filename=${iteration}"th_mig_"  
  matlab -nodesktop -nosplash -noFigureWindows -r \
  "sh_pysit_dir='$pysit_dir',sh_input_dir='$vel_dir',\
  sh_nz=$nz,\
  sh_nx=$nx;\
  CNN_RWI_prepare_starting_true_vp;quit"
cd ..

# 1.3 Obtain RTM image from the starting velocity model
echo "1.3 Obtain RTM image from the starting velocity model"
cd $pysit_dir
python RTM.py

# 1.4 Prepare the input data for CNN prediction
echo "1.4 Prepare the input data for CNN prediction"
matlab -nodesktop -nosplash -noFigureWindows -r \
"sh_nz=$nz,sh_nx=$nx;CNN_RWI_process_rtm_test_image;quit"
rm -r given*
rm -r out*
mv real_dataset ../CNN_approximation1.0
cd ..

#======================================================================================
# PART 2: Iteratively obtain the predicted velocity model and training dataset
#======================================================================================
# The outer loop of CNN-RWI to iteratively obtain the CNN-predicted 
# velocity model and adaptively create the training velocity mdoel
while [ ${iteration} -le $iter ]
do

  if [ ${iteration} -gt 0 ]
  then
    # 2.0 Slightly smooth the predicted model for model parcellation 
    echo "2.0 slightly smooth the predicted model at "${iteration}" th iteration"
    cd $matlab_dir$vel_dir
    echo "enter into directory "$matlab_dir$vel_dir
    matlab -nodesktop -nosplash -noFigureWindows -r \
    "sh_iter='${iteration}',sh_nz=$nz,sh_nx=$nx;\
    CNN_RWI_train_model;quit" #> 'temp.txt'
    cd ../../
  fi

  # 2.1 Create the training and starting models
  echo "2.1 Create training and starting models at "${iteration}" th iteration"
  cd matlab1.0
    matlab_filename=${iteration}"th_mig_"

    matlab -nodesktop -nosplash -noFigureWindows -r \
    "sh_name='$matlab_filename',\
    sh_ratio=$ratio,sh_nz=$nz,sh_nx=$nx,\
    sh_input_dir='$vel_dir',\
    num_samples=$num_samples,num_threads=$num_threads;\
    CNN_RWI_parcellation;\
    num_samples=$num_samples,num_threads=$num_threads,\
    sh_num_smooth_iteration=$sh_num_smooth_iteration,\
    sh_filter_size=$sh_filter_size,\
    sh_nz=$nz,sh_nx=$nx,sh_pysit_dir='$pysit_dir',\
    sh_water_depth=$sh_water_depth;\
    CNN_RWI_prepare_training_models;\
    quit" #> ${iteration}'matlab_res.txt'

    mv velocity velocity"_"$iteration

  cd ..

  # 2.2 Obtain the training RTM images on PySIT platform
  echo "2.2 Obtain the training RTM images at "${iteration}" th iteration"
  cd $pysit_dir
  for ((i=1;i<$num_threads;i=i+1))
  do
    nohup python RTM.py --data_dir "output"$i --model_dir 'given_models'$i &
  done
  python RTM.py --data_dir "output"$num_threads --model_dir 'given_models'$num_threads
  
  # Set Barrier for thread synchronization
  jobs -l > RTM_run.txt
  i=0
  while [ -s RTM_run.txt ]
  do
    jobs -l > RTM_run.txt
    sleep $sleep_time
  done
  rm RTM_run.txt

  # 2.3 Prepare the dataset for CNN training
  echo "2.3 Prepare the dataset for CNN training at "${iteration}" th iteration"
  matlab -nodesktop -nosplash -noFigureWindows -r \
  "sh_nz=$nz,sh_nx=$nx,\
  num_samples=$num_samples,num_threads=$num_threads;\
  CNN_RWI_process_rtm_train_image;quit" #> 'temp.txt'
  rm -r given*
  rm -r out*
  cp -r train_dataset ../CNN_approximation1.0
  mv train_dataset ${iteration}"train_dataset"
  cd ..

  # 2.4 CNN training (the inner loop)
  echo "2.4 CNN training at "${iteration}" th iteration"
  cd $CNN_dir 
  python ML7_vpFWI.py --max_epochs $CNN_max_epochs \
  --store_weights_freq $store_weights_freq --CNN_num $CNN_last_epochs

  # 2.5 CNN prediction
  echo "2.5 CNN prediction at "${iteration}" th iteration"
  # Output predicted models for next iteration in outer loop  
  python ML7_vpFWI.py --mode real --CNN_num $CNN_last_epochs  
  # Output training results for QC only
  python ML7_vpFWI.py --mode export --CNN_num $CNN_last_epochs 

  # Clean the workspace and store the intermediate results
  echo ${iteration}"th clean the directory for next isteration" 
  mkdir ${iteration}${CNN_result_dir}
  rm -r train_dataset
  mv real_outputs ${iteration}${CNN_result_dir}
  mv train_outputs ${iteration}${CNN_result_dir}
  mv "CNN_weights_vp"$CNN_last_epochs ${iteration}${CNN_result_dir}
  rm -r CNN*
  rm train_dataset
  cp -r ${iteration}${CNN_result_dir}"/CNN_weights_vp"$CNN_last_epochs .

  # 2.6 Output the CNN-predicted model for next iteration
  echo "2.6 Output the CNN-predicted model at "${iteration}" th iteration"
  input_vp_filename=${iteration}${CNN_result_dir}"real_outputs/real0_vp.dat";
  next_iter=`expr ${iteration} + 1`
  output_vp_filename="../"$matlab_dir$vel_dir${next_iter}"th_true_vp"
  
  matlab -nodesktop -nosplash -noFigureWindows -r \
  "sh_input_vp_filename='$input_vp_filename',sh_nz=$nz,sh_nx=$nx,\
  sh_output_vp_filename='$output_vp_filename';output_CNN_predicted_models;quit"
  cd ..

  # 2.7 Increase the iteration number
  echo "2.7 Increase the iteration number at "${iteration}" th iteration"
  let iteration++
done 




























