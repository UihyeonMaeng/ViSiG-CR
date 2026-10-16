"""Minimal editable-install metadata for the ViSiG-CR research repository."""

from setuptools import find_packages, setup


setup(
    name="visig_cr",
    version="2.0.0",
    description=(
        "ViSiG-CR training-free relation-aware visual case graph retrieval "
        "for medical VQA"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    packages=find_packages(include=("visig_cr", "visig_cr.*", "comparison_models", "comparison_models.*")),
)
