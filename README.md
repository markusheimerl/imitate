# control
A learned quadcopter controller

1. make drone see
2. add only sensors and camera as input
3. place treasure chest at target

-> then we can either go on to a new repo "lang"
or try to increase REINFORCE sample efficency by utilizing deepseek r1's thinking abilities



lang would try out transformer architecutre...
but before we do that we actually need proper autodiff in C...

we need better models to increase robustness. We cannot measure k_m and k_f
thats why we need to model to be an RNN and figure this out by itself...
We can supply mass though...

So even before seeing (camera extension board is still a while off)
it needs to stabalize and hover without knowing many of the parameters

lets try to make hovering as stable as possible and focus on improving
simulation accuracy and policy robustness to measure errors,
unknown motor strengths and try to model the real first time
it will start and be embodied. Blind, not knowing many of its own
bodies parameters and configurations. It has to figure those out
by emitting a vector that it consumes again (I think this is how RNNs) work