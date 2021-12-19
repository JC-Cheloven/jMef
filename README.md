# jMef
a 2D Finite Element program

The program performs a finite element analysis of a 2d linear elastic solid.
It is mainly educational software, aiming to be used as a daily tool by
students and teacher in an introductory course on the Theory of Elasticity.

To execute it please launch the file jMef_06.py under Python 3 with the 
libraries NumPy, SciPy & Matplotlib installed. 

There is no need for a user manual:  Instead, some "quick notes" and a 
comprehensive set of emerging "tooltips" have been included in the 
program. This should be enough for someone familiar with FEM methods.

Features:
- Six-node parabolic isoparametric triangular elements are used thoroughly
in the approximation. No further complication here.
- Concentrated and distributed boundary forces, volume forces and thermal 
loading are all supported. The foundation/bearing of the solid can be 
represented by prescribed displacements at points or zones of the
boundary. Also punctual and distributed springs (elastic foundations) are
supported. 
- The intended workflow starts with a discretization provided by the user,
which should be just enough to approximate geometry and boundary conditions.
There is a function "refine" which allows refining the mesh to the desired
level afterwards. A hardcoded example can be run to get the idea. Also,
some files with simple examples are included and ready to load.
