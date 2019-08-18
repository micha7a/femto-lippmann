# Femto-Lippmann
Software for a joined Galatea & LCAV project on digitising Lippmann photography.

---
## Getting stated

### Dependencies

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

### Contribute
If you want to contribute to this repository, you should install 'yapf' via:

    pip install yapf
    
or 
    
    conda install yapf
    
and then run:

    ./scripts/setup_repository

in order to set up `yapf` formatter and git hook for removing non important changes form Jupyter Notebooks.

## TODOs

 - describe models
 - replace old functionality:
    - equivalent patterns
    - adding index of refraction in arbitrary place
 - test different material models
     
     
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
