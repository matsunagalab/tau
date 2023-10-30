#!/bin/bash
#PJM -L rscgrp=lecture-a
#PJM -L gpu=1
#PJM -L elapse=00:15:00
#PJM -g gd71
#PJM -j

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/work/gd71/d71000/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/work/gd71/d71000/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/work/gd71/d71000/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/work/gd71/d71000/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

#python -m openmm.testInstallation
python predict_all.py

