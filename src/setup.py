from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
	install_requires=f.read().splitlines()

setup(
    name='dpatk',
    version='0.0.3',
    description='The Dynamic PET Analysis Toolkit',
    long_description=long_description,
	long_description_content_type='text/markdown',
    author='Corentin Martens',
    author_email='corentin.martens@ulb.be',
	url='https://github.com/cormarte/dpatk.git',
    packages=['.dpatk'],
	classifiers=[
		'Programming Language :: Python :: 3',
		'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
		'Operating System :: Microsoft :: Windows :: Windows 10'
	],
	license='GPLv3',
    include_package_data=True,
    install_requires=install_requires,
    python_requires='>=3.8'
)
