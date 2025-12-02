# Assignment 2: dynamic power system simulation 

The second assignment in the specialisation course module: Methods & algorithms in power systems.


**Task**:

Implement the dynamic model of a droop-controlled converter/classical machine in a power system simulator.

Conduct the simulations on a four bus system with the following data:

| Bus | Type          | V (p.u.) | θ (deg) | P (p.u.) | Q (p.u.) |
|-----|---------------|----------|---------|----------|----------|
| 1   | Slack         | 1.00     | 0.0     |          |          |
| 2   | PV            | 1.00     |         | 1.0      |          |
| 3   | PQ            |          |         | -1.0     | -0.5     |
| 4   | PQ            |          |         | -1.0     | -0.5     |

| Line | From Bus | To Bus | R (p.u.) | X (p.u.) | B (p.u.) |
|------|----------|--------|----------|----------|----------|
| 1    | 1        | 2      | 0.0      | 0.1      | 0.0      |
| 2    | 1        | 3      | 0.0      | 0.2      | 0.0      |
| 3    | 2        | 4      | 0.0      | 0.25     | 0.0      |
| 4    | 3        | 4      | 0.0      | 0.2      | 0.0      |

For the machine/converter model, use parameters from the paper by J. Schiffer et. al. here: https://www.sciencedirect.com/science/article/pii/S0005109814003100.

You may choose freely the software; we support Matlab/Simulink (Gilbert) and Python-based TOPS (Sjur).

**Presentation**:

Present simulation results (plots etc.) to show:

*Preliminaries* 
- Power flow results
- Transient disturbance (e.g. short circuit, line trip, load change etc.)

*Frequency control aspects*:
- Frequency synchronisation
- Droop response/power sharing

You may present this directly in your code (functions and some comments) or in a pdf uploaded to your repo.

------------------------------------

Rules of the game!
- Commit and push, frequently!
- Write good commit messages
- Commit history serves as documentation of your work process
- Push unsuccessful attempts too!
- Follow the AI guidelines as given by the IE-faculty. Don't give your learning away :)
- AI speeds up what you don't want to study (syntax, structure, debugging etc.)
- And also as a learning companion
