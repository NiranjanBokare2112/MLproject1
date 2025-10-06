from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path)->List[str]:
    '''This function will return the list of requirements'''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if "-e ." in requirements:
            requirements.remove("-e .")

setup(
    name="MLproject1",
    author="Niranjan",
    author_email="niranjanbokare2112@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)