from setuptools import find_packages, setup

setup(
    name="parcial_toyota",
    packages=find_packages(exclude=["parcial_toyota_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
