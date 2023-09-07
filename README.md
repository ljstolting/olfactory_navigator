This code instantiates an embodied agent inspired by smell-based navigation in mammals (think blood-hounds, for example). The agent needs to maintain a respiratory rhythm while sampling odorants from its environment and navigating towards the source of the odorant. This is similar to the chemotaxer (Beer et al., ) but different in several key ways
    1. The agent does not have direct access to the concentration of the odorant in its vicinity, but this access is mediated by its sniffing behavior
    2. 

The environment is a two-dimensional grid, with the concentration of odorant at each 

The nervous system of the agent is modeled by a CTRNN

Movement effectors are

Sensors are inspired by 