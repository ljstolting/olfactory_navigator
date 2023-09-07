**Agent and Task**
*Summary*
This code instantiates an embodied agent inspired by smell-based navigation in mammals (think blood-hounds, for example). The agent needs to maintain a respiratory rhythm while sampling odorants from its environment and navigating towards the source of the odorant. This is similar to the chemotaxer (Beer & Gallagher 1992) and models of C. elegans klinotaxis (Izquierdo & Beer, 2013) but different in several key ways
    1. The agent does not have direct access to the concentration of the odorant in its vicinity, but this access is mediated by its sniffing behavior
    2. Sniffing behavior, which modulates the strength of the sensed signal, is (at least by default) independent of movement. 

The environment is a two-dimensional grid, with the concentration of odorant at each point equal to the total concentration divided by the square of the distance (check on this)

The nervous system of the agent is modeled by a CTRNN with N neurons, one of which is an odorant sensor, one of which is a respiration effector, and two of which are motor effectors. This leaves N interneurons.
In the "resp-odor" condition, the respiration effector is (reciprocally?) connected to the odorant sensor
In the "resp-move" condition, the respiration effector is connected to both movement effectors
In the "resp-odor_and_move" condition, the respiration effector is connected to the odorant sensor as well as both movement effectors

Movement effectors are 

Sensors are inspired by

*Relevant Files*
CTRNN.cpp, CTRNN.h, random.h, random.cpp, VectorMatrix.h, FitnessEval.py
**Evolution**
*Summary*
Evolutionary fitness is evaluated by 
    1. Breathing frequency above a certain value
    2. Average proximity to the source of the odorant throughout run

Every agent is evaluated by placing it a close distance away from the odorant source facing it, a close distance away facing away, and a far distance away facing. Fitness is averaged across these three conditions and used to direct the activities of the algorithm.

Genome structure is []
*Relevant Files*
random.cpp, random.h, TSearch.cpp, TSearch.h, VectorMatrix.h
**Analysis**
*Summary*

*Relevant Files*
AgentVisualization.ipynb, StatVisualization.ipynb