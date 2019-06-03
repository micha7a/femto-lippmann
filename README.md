# Femto-Lippmann
Software for a joined Galatea & LCAV project on digitising Lippmann photography.

---
## Getting stated

### Dependencies

This project uses Lippmann project 
(by [Gilles Baechler](http://lcav.epfl.ch/people/Gilles_Baechler)) as a library. 
Clone using:
```
git clone --recurse-submodules 
```

The software is written in `Python 3`, with the specific list of dependencies in 
`requirements.txt`. To install them, run:
```
pip install -r requirements.txt
```

You can also install `Anaconda`, available [here](https://www.anaconda.com).

## TODOs

 - switch to using tmm & scipy.stats (?)
 - material class 
    - how the deposited energy depends on current energy deposited?
    the response is sigmoidal, but how the response translates to deposited energy
 - describe models
 - add layered material
 - replace old functionality:
    - chirps
    - equivalent patterns
    - random phase
    - printing without interference 
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
