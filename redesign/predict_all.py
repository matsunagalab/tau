import pandas as pd
import subprocess

df = pd.read_csv('proteinmpnn.csv')

#for i in range(12):
#for i in range(1020, 1024):
for i in range(len(df)):
    print(i)
    jobname = "seq_" + str(i)
    sequence = df['seq'][i]
    subprocess.run(["/opt/anaconda3/bin/python3", "predict_each.py", jobname, sequence])
    #clear_mem()

