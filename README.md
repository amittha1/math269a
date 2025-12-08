# math269a
Math 269A Quarter-long Exam: Building, Analyzing, and Stress-Testing ODE Solvers
math269a/
|
|-- src/
| |-- solvers.py # Euler, RK2, RK4, BE, TR
| |-- problems.py # Example ODEs (A–H)
| |-- experiments_stage1and2.py # LTE, global error for Euler,  RK2/RK4 order and work–precision
| |-- experiments_stage3.py # Stability, indicators
│ |-- experiments_stage4.py # Implicit solvers on stiff problems
| |-- experiments_stage5.py # Implicit accuracy and Newton iteration counts
| |-- experiments_stage6.py # Stability thresholds and stiff tests
| |-- experiments_stage7.py # Adaptive step-size control (step-doubling, RK23)
| |-- experiments_stage8.py # LMMs (AB2, AM2), zero-stability
|  
|-- figs/ # required and more figures for each stage
|-- reproduce_all.py # Runs every stage and reproduces all plots


## Reproduce plots by running reproduce_all
1. Clone repository
1. cd math269a
2. python reproduce_all.py