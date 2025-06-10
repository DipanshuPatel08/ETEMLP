from setuptools import find_packages, setup
from typing import List

hypen_edot = '-e .'
def get_req(file_path: str) -> List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [request.replace('\n', '') for request in requirements]
        if hypen_edot in requirements:
            requirements.remove(hypen_edot)
    return requirements

setup(
    name="ETEMLP",
    version="0.0.1",
    author="Dipanshu_08",
    author_email='dipanshupatel7999@gmail.com',
    packages=find_packages(),
    install_requires=get_req('requirements.txt')
)
