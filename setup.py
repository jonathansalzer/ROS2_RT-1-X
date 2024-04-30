from setuptools import find_packages, setup

package_name = 'ros2_rt_1_x'

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
    maintainer='jonathan',
    maintainer_email='jonathan@salzer.net',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = ros2_rt_1_x.publisher_member_function:main',
            'rt_target_pose = ros2_rt_1_x.rt_target_pose:main',
            'rt1_real_inference_example = ros2_rt_1_x.models.rt1_real_inference_example:main',
        ],
    },
)
