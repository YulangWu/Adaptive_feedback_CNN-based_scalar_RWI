
Please set max_C as a constant (e.g., 5.0) 
in Line 62:
max_C = 5.5 #max(abs(C.min()), C.max())  # faster than C.abs().max()
in the file:
~/Anaconda3/envs/myenv/lib/python3.7/site-packages/pysit-1.0.1-py3.7-linux-x86_64.egg/pysit/solvers/constant_density_acoustic/time/constant_density_acoustic_time_base.py


