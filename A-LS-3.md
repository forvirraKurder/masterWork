# Assignment LS-3: Conditions for Large Signal Stability of AC Power systems

This assignment follows from the previous one, where we aim to analytically replicate a powerful 
stability result published in 2014 by J. Schiffer et al. in “Conditions for stability of droop-controlled inverter-based
microgrids”, Automatica, Volume 50, Issue 10, October 2014. As mentioned in that previous assignment, 
we are interested in replicating the results that have been presented in section 5 of that article, entitled 
“stability for lossless microgrids”; which in our opinion, is a good starting point for large-signal stability analysis
of power systems. The stability proof detailed in section 5, can be divided into two parts---
which will correspond to the two different assignments. 

Given that some of the details can be arguably considered too complicated, 
you can count on the assistance from Gilbert on Tuesdays from 12h-15h in classroom 265 Sentralbygg 1 every week… 
(or whenever he is free in his office).

**Task**: 

The second part of the proof is to derive the (tuning) conditions such that the energy function H(x) can be considered
as a Lyapunov function, which will be used to conclude on (regional/local) asymptotic stability. 
More precisely, you should prove that:

- $H\left(\bar{x}\right)=0;$
- $H\left(x\right)>0\ \mathrm{\mathrm{in}}\ \ \\{\overline{x}}$	
- $\dot{H}\left(x\right)<0\ \mathrm{\mathrm{in}}\ \ \\{\overline{x}}$
- $\dot{H}\left(\overline{x}\right)=0$ 

Ideally, you should briefly check that the tuning conditions indeed result in a stable grid by directly implementing 
them in your dynamical model.


**Deliverables**: 

Your notes of the mathematical derivation, which you should be able to defend in the oral session.