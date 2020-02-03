from setuptools import setup

setup(
   name='astrokit',
   version='1.0',
   description='toolkit for astro data ',
   author='Slawa Kabanovic',
   author_email='kabanovic@ph1.uni-koeln.de',
   packages=['astrokit'],  #same as name
   install_requires=['numpy', 'astropy'], #external packages as dependencies
   scripts=[
            'astro/cele',
            'scripts/skype',
           ]
)
