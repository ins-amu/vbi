.. raw:: html

   <link rel="stylesheet" type="text/css" href="_static/custom.css">

Inference Methods & Examples
=============================

VBI provides multiple approaches for parameter inference, each with different strengths and use cases.

Inference Method Overview
-------------------------

.. list-table:: Inference Methods Comparison
   :header-rows: 1
   :class: color-caption

   * - **Method**
     - **Dependencies**
     - **Performance**
     - **Best For**
   * - **CDE (Conditional Density Estimation)**
     - NumPy + autograd
     - Fast, lightweight
     - Quick inference, limited resources
   * - **SBI (Simulation-Based Inference)**
     - PyTorch + sbi
     - State-of-the-art
     - Complex problems, research

CDE-Based Inference (Lightweight)
----------------------------------

**Conditional Density Estimation** using pure NumPy implementation - no PyTorch required.

**Key Features:**

- ✅ **Lightweight**: Only requires NumPy and autograd
- ✅ **Fast**: Efficient implementation for moderate-scale problems  
- ✅ **Transparent**: Clear mathematical foundation
- ✅ **Flexible**: Easy to customize and extend

**Available Methods:**

.. toctree::
   :maxdepth: 1

   examples/inference_cde_mdn_basic
   examples/inference_cde_maf_basic
   examples/damp_oscillator_cde

**When to use CDE:**

- Limited computational resources
- Want to avoid heavy PyTorch dependency
- Need transparent, interpretable inference
.. - Working with moderate-dimensional problems (< 20 parameters)

SBI-Based Inference (Advanced)
-------------------------------

**Simulation-Based Inference** using PyTorch and the sbi library - state-of-the-art methods.

**Key Features:**

- 🚀 **State-of-the-art**: Latest neural density estimation techniques
- 🧠 **Scalable**: Handles high-dimensional problems efficiently
.. - 🔬 **Research-grade**: Used in cutting-edge neuroscience research
- 🔧 **Comprehensive**: Multiple inference algorithms (SNPE, SNLE, SNRE)

**Current Examples:**

*Note: Most simulation examples currently use SBI for inference. We're working on separating these into dedicated inference tutorials.*

**When to use SBI:**

- Working with high-dimensional parameter spaces (> 10 parameters)
- Need state-of-the-art performance
- Complex, multimodal posterior distributions

.. Choosing Your Inference Method
.. ------------------------------

.. **Decision Tree:**

.. 1. **Do you have PyTorch and sufficient computational resources?**
   
..    - **Yes** → Consider SBI methods
..    - **No** → Use CDE methods

.. 2. **How many parameters are you estimating?**
   
..    - **< 10 parameters** → CDE is often sufficient
..    - **> 10 parameters** → SBI recommended

.. 3. **What's your priority?**
   
..    - **Speed of setup** → CDE
..    - **State-of-the-art accuracy** → SBI
..    - **Interpretability** → CDE
..    - **Scalability** → SBI

Getting Started
---------------

**For CDE Inference:**

1. Start with :doc:`examples/inference_cde_mdn_basic` - Learn MDN basics
2. Try :doc:`examples/inference_cde_maf_basic` - More advanced MAF method
3. Apply to your brain model of choice

**For SBI Inference:**

1. Install full VBI: ``pip install vbi[inference]``
2. Check existing simulation examples that include SBI
3. Adapt to your specific use case

Upcoming Examples
-----------------

We're working on expanding the inference examples:

- **CDE + Brain Models**: Direct application to Jansen-Rit, Wilson-Cowan models
- **SBI Tutorials**: Dedicated SBI workflow examples
- **Method Comparisons**: Side-by-side CDE vs SBI comparisons
- **Real Data Examples**: Using experimental neuroimaging data

Performance Guidelines
----------------------

**CDE Performance Tips:**

- Use appropriate number of mixture components (MDN) or layers (MAF)
- Monitor training convergence carefully
- Consider data preprocessing/normalization

**SBI Performance Tips:**

- Use GPU acceleration when available
- Tune simulation budget vs accuracy trade-off
- Consider sequential vs single-round inference

**Memory and Speed:**

- **CDE**: Lower memory usage, faster setup
- **SBI**: Higher memory usage, potentially faster inference for complex problems

Contributing
------------

We welcome contributions of new inference examples! See our `CONTRIBUTING.md` for guidelines on adding new tutorials and examples.
