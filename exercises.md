# Exercises for Day 4
Cython, MPI parallelization, plotting with matplotlib, testing and documentation of code

## 1. Speed optimization using Cython

#### a. As a simple start, try to reproduce the ```primes.py``` Cythonize example from the lecture notes and familiarize yourself on how to use Cython
Check folder Cython
#### b. Take a look at the example ```rbf.py```<sup>[1](#myfootnote1)</sup> which uses Gaussian Radial Basis Functions (RBFs) as an approximation scheme for some given data. 
How much speed up do you gain by using existing Python packages like Scipy? (Feel free to use your knowledge about performance testing 
and improve the current way of timing the different implementations)
0.07 (Scipy) vs 5.8 (naive python) seconds
#### c. In the above example, why is Scipy faster than the naive Python implementation? 
Which part of the Python code is slowing things down? (Again, use the profiling skills you learned earlier)
Last for loop in naive python implementation takes most hits and time

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     6                                           @profile
     7                                           def rbf_network(X, beta, theta):
     8                                           
     9         1          6.0      6.0      0.0      N = X.shape[0]
    10         1          1.0      1.0      0.0      D = X.shape[1]
    11         1          6.0      6.0      0.0      Y = np.zeros(N)
    12                                           
    13      1001        492.0      0.5      0.0      for i in range(N):
    14   1001000     466482.0      0.5      3.6          for j in range(N):
    15   1000000     421262.0      0.4      3.3              r = 0
    16   6000000    3061859.0      0.5     23.7              for d in range(D):
    17   5000000    6609754.0      1.3     51.1                  r += (X[j, d] - X[i, d]) ** 2
    18   1000000     597314.0      0.6      4.6              r = r**0.5
    19   1000000    1782858.0      1.8     13.8              Y[i] += beta[j] * exp(-(r * theta)**2)
    20                                           
    21         1          1.0      1.0      0.0      return Y

#### d. How much can you improve the performance of ```rbf.py``` using Cython? 
Use the lecture notes and follow the instructions from the comments

doing the same calculations using Cython it takes
0.05 (Scipy) and 3.98 (naive python) seconds
In [1]: import rbf
('Python: ', 3.9838287830352783)
('Scipy: ', 0.052184104919433594)
## 2. MPI parallelization

#### a. Write a simple MPI script ```mpi_ranks.py``` that prints the rank of the different processes when running 
```
mpirun python mpi_ranks.py
```

Running into the following error code when executing the command:
Fatal error in MPI_Init_thread: Invalid group, error stack:

when trying it on the davinci server, I don't seem to be able to install the necessary package there:
-bash-4.2$ python -m pip install mpi4p
/usr/bin/python: No module named pip
-bash-4.2$ conda install -c conda-forge mpi4pi
-bash: conda: command not found
#### b. Write a small script ```mpi_sum.py``` which calculates the sum over all ranks and prints the result from the process with rank 0.
Hint: Have a look at the tutorials from the mpi4py documentation page: [https://mpi4py.scipy.org/docs/usrman/tutorial.html](https://mpi4py.scipy.org/docs/usrman/tutorial.html)

## 3. Plotting with matplotlib

#### a. Download the jupyter notebook [customized_plotting.ipynb](customized_plotting.ipynb) and try to understand the individual steps in creating a plot and customizing it
Done

#### b. Produce your own data (curves, images, point clouds, etc...) and create a nice plot. Get inspired by the [matplotlib gallery](https://matplotlib.org/gallery/index.html) and use the extensive documentation of the [pyplot API](https://matplotlib.org/api/pyplot_summary.html)
added 'own_customized_plot.ipynb' to the repository

## 4. Testing code with py.test

#### a. Install pytest using e.g.
```
pip install pytest --user
```

#### b. Write a module ```test_simple_math.py``` which tests the correctness of the math operations inside ```simple_math.py```.
You can run the tests using the ```py.test``` command. Note that the name of your test functions should start with ```test_```.

## 5. Documenting code

#### a. Write some meaningful docstrings for the ```simple_math.py``` module, e.g. following the [numpy_doc](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt) standard.

#### b. Create a documenation html page for the ```simple_math.py``` module using Sphinx.
Follow the instructions from the lecture notes.


********************************************************************************************************
<a name="myfootnote1">1</a>: Taken from [http://nealhughes.net/cython1/](http://nealhughes.net/cython1/)
