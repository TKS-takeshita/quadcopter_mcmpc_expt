from setuptools import setup

package_name = 'px4_response_nn'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools', 'torch', 'pandas'],
    zip_safe=True,
    maintainer='you',
    description='PX4 response NN',
    entry_points={
        'console_scripts': [
            'train_px4_response_nn = px4_response_nn.train:train',
        ],
    },
)