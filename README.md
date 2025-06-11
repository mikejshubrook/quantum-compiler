# quantum-compiler
Quantum compilation algorithm to decompose arbitrary two qubit gates into the native gate set of different quantum computing architectures.

A video demonstration of the old version of the code (previous Qiskit version), as well as an explanation of the quantum compiler theory behind it, can be found in this video on my YouTube channel: https://www.youtube.com/watch?v=NSC6NrH7Y-g&t=753s&ab_channel=MikeShubrook

This code has been programmed to work for a trapped-ion quantum computer, and therefore depends on the two-qubit basis gate being the Molmer-Sorensen gate, and the Euler basising being XYX. This should be changed if the compiler is to be used for a different architecture.
There are also hardware-specific noise parameters hard coded in. These should be changed for different architectures. 

This is an ongoing project that I plan to keep working on, so there may be some parts of the code that have TODOs or comments giving ideas on what needs to be done next.


