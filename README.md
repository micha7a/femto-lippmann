# Femto-Lippmann

Simulations of propagation of light through non-uniform media. Use transfer-matrix method for layered materials.

Simulations originally developed for a joined Galatea & LCAV project on digitising Lippmann photography. 

---
## Getting stated

### Dependencies & Installation

The software is written in `Python 3`, with the specific list of dependencies in 
`requirements.txt`. To install them, run:
```
pip install -r requirements.txt
```

You can also install `Anaconda`, available [here](https://www.anaconda.com), 
and run:
```
conda install --file requirements.txt
```

### Documentation

The functions and objects are documented in the files than define them. If you
want to see the documentation from Python console, you can run:

    import wave
    help(wave)
    help(weve.PlanarWave)
    help(wave.PlanarWave.plot)
    
to see documentation on the whole module, object and function, respectively.

### Examples
Many examples how to use this code are in the `Examples.ipnb`. In order to view 
the notebook, first follow installation instructions above, and then run 
`Jupyter Notebook`. If you used Anaconda on Windows, you can run Jupyter 
Notebook from start menu, but you also always can run it from the terminal. Go 
to directory where this project is located, and run:

    jupyter notebook

It will open your browser (or a new tab in your browser), where you will be 
able to se the list of the files in this project. You can then click on any 
`Examples.ipnb` file to see the interactive examples. In order to run code in a 
cell of Jupyter Notebook, press `shift + enter` or pick "run cell" from the 
menu on top. 

### Contribute
If you want to contribute to this repository, you should install `yapf` via:

    pip install yapf
    
or 
    
    conda install yapf
    
and then run:

    ./scripts/setup_repository

in order to set up `yapf` formatter and git hook for removing non important 
changes form Jupyter Notebooks.

Before pushing to the project, you should also run unit tests in `tests` 
directory, and adding more tests is always welcome ;)

     
## License

```
Copyright (c) 2019 Michalina Pacholska

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
