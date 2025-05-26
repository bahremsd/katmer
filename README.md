# **katmer: inverse design of optical thin films & material fitting with jax and equinox**

<div align="center">
  <a href="https://pypi.org/project/katmer/">
    <img src="https://github.com/bahremsd/katmer/blob/master/docs/images/logo_katmer_v1.png" alt="katmer">
  </a>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#documentation">Documentation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#database">Database</a></li>
    <li><a href="#benchmarks">Benchmarks</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#credits">Credits</a></li>
    <li><a href="#contact-and-support">Contact and Support</a></li>
  </ol>
</details>


## Introduction

`katmer` is a high-performance, research-grade python library purpose-built for the inverse design of optical multilayer thin films. Bridging classical optics with modern computational tools, `katmer` offers a comprehensive suite of features tailored for photonic engineers, researchers, and designers aiming to optimize thin-film structures with precision, speed, and flexibility.

In traditional multilayer thin-film design, achieving desired optical responses—such as reflectance, transmittance, or absorbance—often requires a trial-and-error approach or time-consuming simulations. `katmer` redefines this workflow by integrating gradient-based optimization into the design loop. This allows users to arrive at precise, physically valid solutions in a fraction of the time, enabling faster iteration and deeper insights.

At its core, `katmer` is powered by two cutting-edge libraries:

- [`tmmax`](https://github.com/bahremsd/tmmax): A transfer matrix method (TMM) engine implemented with JAX, enabling just-in-time (JIT) compilation, vectorized operations, and XLA (Accelerated Linear Algebra). This dramatically speeds up the evaluation of optical properties across complex multilayer systems.

- [`equinox`](https://github.com/patrick-kidger/equinox): A lightweight yet powerful neural network library designed to work seamlessly with JAX. Its pytree-based architecture enables flexible model building and efficient parameter management, critical for embedding neural architectures into the optimization process.


<div align="center">
  <a href="https://pypi.org/project/katmer/">
    <img src="https://github.com/bahremsd/katmer/blob/master/docs/images/readme_info.png" alt="katmer">
  </a>
</div>

### Core Capabilities

#### Optimum Number of Layers Prediction

One of `katmer`’s key features is the intelligent determination of the optimal number of layers required to achieve a user-defined target response. This is achieved using a pre-trained equinox model, which maps spectral objectives (across wavelength and incidence angles) to an ideal layer count—preventing over-engineering while minimizing fabrication complexity and cost. This forms a robust foundation for downstream optimization.

#### Thickness Sensitivity-Aware Inverse Design

Once the layer number is established, `katmer` performs gradient-based inverse design to determine the optimal thickness of each layer. Crucially, this process incorporates deposition uncertainties, modeling real-world experimental errors by simulating the effect of layer thickness deviations. This leads to designs that are not only optimal in simulation but also robust in practice.


#### Inverse Design with Material Constraints

Real-world thin-film fabrication is governed by material compatibility constraints—not all materials can be deposited on each other. `katmer` addresses this by constructing a material compatibility graph, currently available for materials supported in `tmmax`. During inverse design, this graph is used to ensure only physically realizable material stacks are considered. Furthermore, each material comes with its own deposition thickness bounds, which `katmer` rigorously respects during optimization.


#### Tickness-only Optimization

If you already have a fixed material distribution and only wish to optimize the layer thicknesses, `katmer` accommodates this effortlessly. With gradient-based methods, the solver converges rapidly, delivering optimal thickness distributions within the specified physical constraints.

#### Optical Property Reconstruction from Experimental Data

`katmer` also enables optical property reconstruction: if your multilayer system contains a material with unknown refractive index (n) or extinction coefficient (k), and you possess experimental spectral data, `katmer` can infer the n and k values through inverse modeling. This feature is particularly useful for characterizing novel or proprietary materials used in your thin-film designs.

## Documentation

The complete documentation for `katmer` is available in the [Example Gallery](https://github.com/bahremsd/katmer/tree/master/docs/examples) within the `docs` directory. This repository provides an extensive set of examples demonstrating the key functionalities of `katmer`, enabling users to efficiently analyze and manipulate multilayer thin-film stacks.

## Usage

## Database

## Benchmarks

## Installation

You can install `katmer` via PyPI:

```bash
pip3 install katmer
```

## License

This project is licensed under the [MIT License](https://opensource.org/license/MIT), which permits free use, modification, and distribution of the software, provided that the original copyright notice and license terms are included in all copies or substantial portions of the software. For a detailed explanation of the terms and conditions, please refer to the [LICENSE](https://github.com/bahremsd/katmer/blob/master/LICENSE) file.

## Credits

Also if you find the `katmer` library beneficial in your work, we kindly ask that you consider citing us.

```bibtex
@software{katmer,
  author = {Bahrem Serhat Danis, Esra Zayim},
  title = {katmer: inverse design of optical thin films and material fitting with jax and equinox},
  version = {1.0.0},
  url = {https://github.com/bahremsd/katmer},
  year = {2025}
}
```

## Contact and Support

For any questions, suggestions, or issues you encounter, feel free to [open an issue](https://github.com/bahremsd/katmer/issues) on the GitHub repository. This not only ensures that your concern is shared with the community but also allows for collaborative problem-solving and creates a helpful reference for similar challenges in the future. If you would like to collaborate or contribute to the code, you can contact me via email.

Bahrem Serhat Danis - bdanis23@ku.edu.tr
