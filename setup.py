from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('License.md') as f:
    license = f.read()

setup(
    name='bath_fitting',
    version='0.1.0',
    description='Bath fitting via semi-definite relaxation package',
    long_description=readme,
    author='Leonardo Zepeda-Nunez',
    author_email='zepedanunez@wisc.edu',
    url='https://github.com/Forgotten/BathFitting',
    license=license,
    install_requires=['numpy', 'scipy', 'cvxpy']
    packages=setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3",
                "License :: MIT License",
                "Operating System :: OS Independent",],
)