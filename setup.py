from setuptools import setup, find_packages

# pip install numpy==1.23.0 or less

setup(
    name='astrokit',
    version='0.1',
    description='toolkit for astro data ',
    author='Slawa Kabanovic',
    author_email='kabanovic@ph1.uni-koeln.de',
    packages=find_packages(exclude=('doku')),  # same as name
    # external packages as dependencies
    install_requires=['numpy',
                      'astropy',
                      'astroquery',
                      'PyAstronomy',
                      #'aplpy',
                      'scipy',
                      'sympy',
                      'astrodendro',
                      'matplotlib',
                      'pandas',
                      'scikit-learn',
                      'joblib',
                      'reproject',
                      'IPython',
                      #'pomegranate',
                      'cygrid'],
    # scripts=[
    #          'astro/celestial.py',
    #          'math/curve.py',
    #          'math/transform.py',
    #          'num/solver.py',
    #          'prism/dendrokit.py',
    #          'prism/mask.py',
    #          'prism/specube.py',
    #          'astrokit/watch/progress.py'
    #         ]
    zip_safe=False
)
