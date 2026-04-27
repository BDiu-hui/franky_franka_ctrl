from setuptools import setup

package_name = "franky_franka_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/franky_http.launch.py"]),
        (
            f"share/{package_name}/scripts",
            [f"{package_name}/franky_http_server.py"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="lumos",
    maintainer_email="lumos@example.com",
    description="HTTP bridge for precise Franka control using franky and Ruckig trajectory generation.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "franky_http_server.py = franky_franka_control.franky_http_server:main",
        ],
    },
)
