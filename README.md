# Parallel Floyd Warshall

Parallel Floyd-Warshall<br/>
Cuda compilation tools, release 11.0, V11.0.221<br/>
<br/>
Compile with `nvcc example.cu`.
<br/><br/>
Run examples with <br>
`a.exe FILE_PATH CPU_THREADS [PRINT_RESULT]` <br/>
eg. <br>
`a.exe Examples\e1.txt 10`  <br>
`a.exe Examples\e1.txt 10 true`
<br/><br/>
or randomly generate a N x N distance matrix with <br>
`a.exe random N CPU_THREADS [PRINT_RESULT]` <br/>
eg. <br>
`a.exe random 5000 10` <br>
`a.exe random 5 10 true` <br>
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
