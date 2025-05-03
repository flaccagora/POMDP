from setuptools import setup, find_packages

setup(
    name="project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pytest",
        "pytest-cov",
        "torch",
        "stable-baselines3",
        "wandb",
    ],
    python_requires=">=3.8",
) 