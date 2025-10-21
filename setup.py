from setuptools import setup, find_packages

setup(
    name="databricks-feature-store-demo",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "databricks-feature-engineering>=0.8.0",
        "mlflow>=2.14.0",
        "scikit-learn>=1.5.0",
        "pandas>=2.1.0",
        "numpy>=2.0.0"
    ],
    python_requires=">=3.12"
)