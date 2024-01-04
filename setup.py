from setuptools import setup, find_packages

# exec(open("ssdn/version.py").read())

setup(
    name="cet_pick",
    version="0.1",  # noqa
    packages=find_packages(),
    install_requires=[
        "nptyping",
        "h5py",
        "imagesize",
        "overrides",
        "colorlog",
        "colored_traceback",
        "tqdm",
        "pyYAML",
        "mrcfile",
    ],
)
