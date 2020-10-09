# Parallel Floyd Warshall

Parallel Floyd-Warshall<br/>
Cuda compilation tools, release 11.0, V11.0.221<br/>
<br/>
Compile with `nvcc example.cu`.
<br/><br/>
Run examples with <br>
`a.exe <filepath> <CPU_THREADS> <print_result (default = false)>` <br/>
eg. <br>
`a.exe Examples\e1.txt 10 false` 
<br/><br/>
n = 5000: <br/>
Time elapsed (GPU): 9.842497 <br/>
Time elapsed (CPU, multithread): 90.444 <br/>
Time elapsed (CPU; single thread): 335.686 <br/>
<br/><br/>
n = 1000: <br/>
Time elapsed (GPU): 0.277910 <br/>
Time elapsed (CPU, multithread): 1.546 <br/>
Time elapsed (CPU; single thread): 2.979 <br/>
<br/><br/>
n = 500: <br/>
Time elapsed (GPU): 0.051269 <br/>
Time elapsed (CPU, multithread): 0.639 <br/>
Time elapsed (CPU; single thread): 0.381 <br/>
