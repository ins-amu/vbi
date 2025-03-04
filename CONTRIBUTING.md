# Contributing to VBI

Thank you for your interest in contributing to the VBI project! We welcome contributions in the form of bug reports, feature requests, and code contributions. Please follow the guidelines below to ensure a smooth contribution process.

## Reporting Issues

If you encounter any bugs or have feature requests, please open an issue on our [GitHub repository](https://github.com/ins-amu/vbi/issues). Provide as much detail as possible, including steps to reproduce the issue and any relevant logs or screenshots.

## Code Contributions

### Getting Started

1. Fork the repository on GitHub.
2. Clone your forked repository to your local machine:
    ```sh
    git clone https://github.com/your-username/vbi.git
    cd vbi
    ```
3. Install the required dependencies:
    ```sh
    pip install .
    # For development, install additional dependencies:
    pip install -e .[all,dev,docs]
    ```

### Making Changes

1. Create a new branch for your feature or bugfix:
    ```sh
    git checkout -b my-feature-branch
    ```
2. Make your changes in the codebase.
3. Ensure your code follows our coding standards and includes tests for any new functionality.
4. Run the tests to verify your changes:
    ```sh
    pytest
    ```

### Submitting a Pull Request

1. Push your changes to your forked repository:
    ```sh
    git push origin my-feature-branch
    ```
2. Open a pull request on the original repository. Provide a clear description of your changes and any relevant issue numbers.

### Code Review

Your pull request will be reviewed by one of the project maintainers. Please be responsive to any feedback and make necessary changes. Once your pull request is approved, it will be merged into the main branch.

## Coding Standards

- Follow PEP 8 for Python code style.
- Write clear and concise commit messages.
- Include docstrings for all functions and classes.
- Write tests for any new functionality or bugfixes.

## Documentation

If you are contributing to the documentation, make sure to update the relevant `.rst` files in the `docs` directory. You can build the documentation locally using:
```sh
cd docs
make html
```