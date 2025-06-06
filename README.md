# Near Inertial Oscillations Reconstruction

![alt text](https://github.com/HugoJacq/NIO_reconstruct/blob/main/-46.5E_40N_t239_centered_on_cyclone..png?raw=true)

Figure 1: Sea state on the 16th of March 2005 0000 UTC, a few days after a intense wind stress peak. Panels (a)-(e): zonal ageostrophic current (m.s−1) outputs from a) the slab model, b) the slab model with the advection term, d) the unsteak model and e) the unsteak model with the advection term. c) is the true ageostrophic current. Panel (f): curl of geostrophic current (s−1).

## The models

The slab model with an advection term:
![alt text](https://github.com/HugoJacq/NIO_reconstruct/blob/main/slab_equation_with_adv.png?raw=true)

The unsteady Ekman model (N layers):
![alt text](https://github.com/HugoJacq/NIO_reconstruct/blob/main/unsteak_2layers_equations.png?raw=true)


## The science

The file `Near Inertial Oscillations Reconstruction Using A Simple Model.zip` is overleaf project where the motivation / methods and first results are gathered.
Look into `experiment` to find the scientific question and what the code Idid to answer them.

## Environment

create a conda env, then install using conda packages listed in `packages_list_conda.txt` then install using pip packages listed in `packages_list_pip.txt`.

Be carefull if a new version of any of the JAX packages is up, make sure to update them together. Theses librairies evolve at a fast rate and incompatibilities happen rather sooner than later.

## License

MIT License, see LICENCE.txt
