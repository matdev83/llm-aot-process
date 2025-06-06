from setuptools import setup, find_packages

setup(
    name="llm-aot-process",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "openai",
        "llm-accounting",
    ],
)
