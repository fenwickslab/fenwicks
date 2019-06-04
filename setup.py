from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ["numpy", "scipy", "scikit-learn",
                "librosa",
                "pandas",
                "matplotlib", "plotly", 'seaborn',
                "typing",
                "beautifulsoup4",
                "tqdm",
                "lightgbm",
                ]

setup(
    name='fenwicks',
    packages=find_packages(),

    include_package_data=True,

    install_requires=requirements,
    python_requires='>=3.6',

    description='Fenwicks',
    long_description=readme,
    long_description_content_type='text/markdown',
    keywords='deep learning, machine learning',

    license="GNU GPL",

    url='https://github.com/fenwickslab/fenwicks',

    author="David Yang",
    author_email='yin@yang.net',

    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
