#This is a wrapper for launching different retrivals for Stellar Contamination project
#Author: Prajwal Niraula
#Insitute: MIT
#Platform: MIT_Supercloud

import numpy as np
import glob
import os

BaseText="#!/bin/bash\n#SBATCH -c 4 -n 1\n#SBATCH -o Run_%j.log\nsource /etc/profile\nmodule load anaconda/2023a\nexport OMP_NUM_THREADS=4\nexport MKL_NUM_THREADS=4\npython retrieval_supercloud.py INPUTFILE NCOMPONENTS"

Targets = glob.glob("data/*.csv")

JobNum = 0
for Target in Targets:
    for N in range(1,5):
        print("N:", N)
        ReplacedText = BaseText.replace("INPUTFILE", Target).replace("NCOMPONENTS", str(N))
        print("The replaced text is given by:", ReplacedText)
        #print
        LauncherName = "SubmitJob_"+str(JobNum).zfill(5)+".sh"
        print(LauncherName)
        with open(LauncherName, 'a') as f:
            f.write(ReplacedText)
        os.system("chmod u+x %s" %LauncherName)

        JobNum+=1
        os.system("LLsub %s" %LauncherName)
        

