# NeRFAcc MindSpore
This folder contains the naive (directly functionally translated) MindSpore implementation of the nerfacc [occupancy grid estimator](https://www.nerfacc.com/methodology/sampling.html#occupancy-grid-estimator). With the features of:
* No c (cuda/cann) extentions needed;
* Functionality correctness,

the implementation here can be plugged into our current MVDream-threestudio pipeline for the exactly same rendering behavior as the original repo. However, the caveat is that it will be pretty slow as such an operator based implementation is not as optimized as the c extention based implementation, from the cuda/cann api perspectives.

You may check the functionality correctness by running the unit tests under [threestudio_tests/nerfacc](../../../tests/threestudio_tests/nerfacc).
