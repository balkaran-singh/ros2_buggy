from setuptools import find_packages, setup

package_name = 'buggy_core'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='buggy',
    maintainer_email='buggy@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'motor_node = buggy_core.motor_node:main',
            'qos_relay = buggy_core.qos_relay:main',
            'brain_node = buggy_core.brain_node:main'
        ],
    },
)
