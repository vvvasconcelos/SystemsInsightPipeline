from setuptools import setup, find_packages

setup(
    name='sip_systemsinsightpipeline',
    version='0.3.1',
    description=(
        'Systems Insight Pipeline (SIP): convert causal loop diagrams from Kumu '
        'Excel exports into computational system dynamics models; simulate, '
        'optimize, and analyze interventions under uncertainty.'
    ),
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/vvvasconcelos/SystemsInsightPipeline',
    author='Vítor V. Vasconcelos',
    author_email='v.v.vasconcelos@uva.nl',
    license='GPL-3.0-or-later',
    python_requires='>=3.9',
    packages=find_packages(exclude=('tests',)),
    install_requires=[
        'numpy',
        'pandas>=1.3.0',
        'scipy>=1.7',
        'matplotlib>=3.5',
        'seaborn>=0.11.0',
        'networkx>=2.5',
        'openpyxl>=3.1.0',
        'tqdm',
        'tabulate>=0.8.9',
        'SALib>=1.4',
        'scikit-learn>=1.0',
    ],
    extras_require={
        'dev': [
            'pytest',
            'nbformat',
            'nbclient',
            'jupyter',
            'ipykernel',
        ],
    },
)
