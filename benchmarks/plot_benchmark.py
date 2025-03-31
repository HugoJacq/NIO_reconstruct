import os
import matplotlib.pyplot as plt

result_local = 'benchmark_results.txt'
result_jackz = 'benchmark_results_jackz.txt'

L_results = [result_local,result_jackz]

# PLOT
dpi=200
path_save_png = './png_benchmark/'

os.system('mkdir -p '+path_save_png)


nbparams = {}
tforward = {}
tcost = {}
tgrad = {}

# getting data
with open(result_local, "r") as f:
    line = f.readline()
    i = 0
    stop = False
    start = False
    while (line) and not stop:
    
        line = f.readline()    
        if line[:3]=='*==':
            stop = True
            continue
        if line[:4]=='* C3':
            istart = i+1
            
        if 'istart' in locals():
            if istart==i:
                start = True

        if start and (not line[:4]=='   -'):
            
            myinfo = line.strip().split(',')
            current_name = myinfo[0]
            nbparams[current_name] = myinfo[1]
            
        elif start and (line[:5]=='   - '):
            
            myinfo = line[5:].strip().split(',')
            if myinfo[0]=='forward':
                tforward[current_name] = float(myinfo[1])
            elif myinfo[0]=='cost':
                tcost[current_name] = float(myinfo[1])
            elif myinfo[0]=='gradcost':
                tgrad[current_name] = float(myinfo[1])
        i=i+1
            
# plotting


