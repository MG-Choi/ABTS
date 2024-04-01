from setuptools import setup, find_packages

setup(
    name = 'ABTS',
    version = '0.0.3',
    description = "algorithm for Agent Based Travel Scheduler",
    url = 'https://github.com/MG-Choi/ABTS',
    author = 'MoongiChoi',
    author_email = 'u1316663@utah.edu',
    packages = find_packages(),
    package_data = {'ABTS': ['data/cbg_milwaukee.cpg', 'data/cbg_milwaukee.dbf', 'data/cbg_milwaukee.prj', 'data/cbg_milwaukee.sbn',
                            'data/cbg_milwaukee.sbx', 'data/cbg_milwaukee.shp', 'data/cbg_milwaukee.shp.xml', 'data/cbg_milwaukee.shx',
                            'data/prob_2020_09_combined.xlsx', 'data/repaired_NHTS.csv', 'data/trip_mode_prop_all.csv']},
    include_package_data = True,
    install_requires = ['tqdm',
                        'numpy',
                        'pandas', 
                        'shapely>=2.0.2', 
                        'pyproj>=3.3.0', 
                        'geopandas>=0.14.0', 
                        'networkx>=3.2.1', 
                        'osmnx>=1.7.1', 
                        'geopy>=2.2.0' ]
)







'''
note: How to make library
- 모두 seqC -> py로 저장.

- cmd (administrator) -> cd repository
- python setup.py sdist bdist_wheel

- 이후 upload를 위해 https://pypi.org/manage/account/token/ 여기서 token을 받아야함. 그리고 밑에 처럼 토큰을 입력.
- 예로 토큰이 pypi-asdadsdas-adwdas 라면
- twine upload dist/* -u __token__ -p pypi-asdadsdas-adwdas
- 업데이트시에는 setup.py -> 0.02로 하고 다시 위 과정 반복


library test는 cmd에서 한다.

- pip uninstall 
- pip install sequentPSS


* 주의할 점:
random이나 os와 같이 깔려있는 library의 경우 위에 install_requires에 쓰지 않는다. py안에 바로 import로 쓰면 된다.

'''


#repository: C:\Users\MoongiChoi\Desktop\MG\양식, 코드 등\Python\Library\indoorCont

#참고:https://lsjsj92.tistory.com/592
#https://developer-theo.tistory.com/entry/%ED%8C%8C%EC%9D%B4%EC%8D%AC-GitHub-Repository-%EC%83%9D%EC%84%B1%EB%B6%80%ED%84%B0-PyPI-Package-%EB%B0%B0%ED%8F%AC%EA%B9%8C%EC%A7%80

#위에서 버전 문제 발생: !pip install --upgrade requests
