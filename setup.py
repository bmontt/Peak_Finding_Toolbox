from setuptools import setup, find_packages

setup(
    name='Peak_Finding_Toolbox',
    version='0.1.0',
    description='Adaptive peak‐detection toolbox for ABR and HRIR signals',
    author='Brody Montag',
    packages=find_packages(),  # will pick up the `toolbox/` package
    install_requires=[
        'numpy>=1.18.0',
        'scipy>=1.4.0',
        'pandas>=1.0.0',
        'mne>=1.2.0',
        'mne-bids>=0.8.0',
        'python-sofa>=0.2.0',    # or `pysofaconventions` if you prefer
        'matplotlib>=3.0.0',
        'tqdm>=4.0.0'
    ],
    entry_points={
        'console_scripts': [
            'abr_toolbox=toolbox.cli:cli',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Instrumentation',
    ]
)
