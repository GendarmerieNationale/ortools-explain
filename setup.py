from setuptools import setup, find_packages

setup(
    name="ortools_explain",
    version="1.0.0",
    description="Package encodant différentes classes et fonctions utiles pour compléter la libraire ortools",
    author="ST(SI)2 / Datalab",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires="~=3.7",
    install_requires=["ortools>=9.3"]
)
