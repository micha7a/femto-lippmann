# Femto-Lippmann
Software for a joined Galatea & LCAV project on digitising Lippmann photography.

---
# Getting stated

## Dependencies

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
    - propagation of wave:
    seems best to use now Gilles matrix theory (still).
     Wee need spectrum at each z, might use different discretization for simulation 
     and for plotting. The equation we want to calculate is:
     $$\sum_{k} |A_+(k, z) + A_-(k, z)|^2 dk $$
