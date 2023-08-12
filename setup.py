from setuptools import setup, find_packages

setup(
    name = 'artemis',
    version='0.0.7',
    description='Personal library to compute and draw',
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
    url ='https://github.com/dowoonlee/artemis.git',
    author='dwlee',
    author_email='dwlee717@gmail.com',
    license='artemis',
    packages = ["artemis", "artemis.datagenerator", "artemis.myplot", "artemis.stats", "artemis.util"],
    zip_safe=False,
    install_requires=[
        'numpy==1.22.3',
        'matplotlib==3.5.2',
        'astropy==5.1',
        'scipy==1.7.3',
        'pandas==1.4.3',
        'imageio==2.19.3'
    ]
)