# ABTS
Agent Based Travel Scheduler library to predict individual travel schedules


## License
ABTS / version 0.0.3
- install:

```python
!pip install ABTS
```

###### 이 알고리즘은 개개인의 travel schedule을 예측하는 시뮬레이션으로, ABTS (Agent-Based Travel Scheduler)이다. 파라미터 4가지를 이용하여 예측을 하며, 기존의 OD데이터를 이용하여 과거의 travel을 trip purpose, age 별로 decompose할 수 있고, 미래 예측 역시 가능하다. 모델에 대한 더 자세한 description은 다음의 두 논문에서 찾을 수 있다.

- Choi, M., & Hohl. A. (2024). Derivation of Spatiotemporal Risk Areas and Travel Behaviors During Pandemic Through Reverse Estimation of Mobility Patterns by Agent-Based Modeling, Spatial and Spatio-temporal Epidemiology. Under review (1st round)

- Choi, M., Seo, J., & Hohl. A. (2024). ABTS: Agent-Based Travel Scheduler. In process


###### Figure 1 shows the framework of ABTS. ABTS utilizes a three-step process to generate the initial outcome of individual travel schedules: 1) Comprehensive Travel Classifier, 2) Land-Use Estimator, and 3) Individual Travel Schedule. The output of ABTS includes individual daily trip chains that consists of each trip purpose, occurrence time, destination, duration, and mode, tailored according to age group.

![Figure1. Framework_of_ABTS](/ABTS/image/Figure1_Framework_of_ABTS.png)

###### Comprehensive travel classifier에서는 Table 1과 같이 trip purpose와 trip mode, Age group을 각각의 기준으로 classify한다. 예로 trip purpose는 9개의 Major trip purpose와 13개의 Sub trip purpose로 나눠진다.

<div style="text-align: center;">
    <img src="/ABTS/image/Classification_method_in_ABTS.png" alt="Table 1. Classification in ABTS" width="650"/>
</div>

<div class="center">
    <img src="/ABTS/image/Classification_method_in_ABTS.png" alt="Table 1. Classification in ABTS" width="650"/>
</div>


###### Land-Use estimator에서는 landuse 데이터를 생성한다. 우리는 landuse를 Table 1의 Sub trip purpose와 같이 13개의 category로 구분하여 shp파일로 구축하였다. 데이터의 출처는 밑의 0. Preprocessing 단계에 기술되어 있다.



# Run the model with example data

###### 여기서는 Individual travel schedule generator의 과정을 예제 데이터를 통해 돌려보도록 한다. case area는 Milwaukee County이고 2020년 9월 과거의 travel을 trip purpose별, age별로 decompose하고, 예측하는 과정을 보여주겠다.


## 0. Preprocessing (using sample simulation in library)

###### 여기서는 ABTS모델을 run하기 위해 NHTS data와 SafeGraph data를 preprocessing하는 과정을 보여준다. 원본 데이터의 경우 밑에 기술된 두 홈페이지에서 다운받을 수 있으며, 본 example에서 사용되는 데이터 역시 해당 사이트에서 다운 받은 데이터이다. 하지만 SafeGraph Neighborhood 데이터의 경우, 현재 타 소스에서 제공되고 있는 것으로 보이며 여전히 접근 가능하다.

- NHTS data: 출처는 Federal HighWay Administration (FHWA)이며, 2017년의 National Household Travel Survey 데이터를 사용하였다 (https://nhts.ornl.gov/downloads). 

- SafeGraph data: 출처는 SafeGraph이며 데이터 이름은 Neighborhood patten이다. (https://docs.safegraph.com/docs/neighborhood-patterns). 이 데이터는 한달동안의 한 주를 랜덤으로 하여 한 주 동안의 사람들의 CBG별 이동을 OD로 나타낸 데이터로, 어떤 CBG에서 어떤 CBG로 얼마만큼의 사람이 이동했는지 기록되어있다. 단, trip purpose나 age 등의 정보는 없으며 단지 count만 수집되었다. 우리는 이 데이터를 이용해 확률모델을 만들어 사람들의 trip purpose별 Destination CBG를 선택하는 알고리즘에 적용하였다.


### 0.0. Data import

###### 구글 드라이브에 데이터를 업로드 해 놓았다 (밑 링크). 여기 데이터에는 다음과 같은 데이터가 있다. https://drive.google.com/drive/folders/1YlsNPeTIC-0hjgYNnatuhmIGWQVSvg0-?usp=sharing 

- trippub.csv: NHTS data (2017)
- Milwaukee_parcels.shp: Milwaukee landuse data
- neighbor_2020_09.csv: SafeGraph Neighborhood data in Milwaukee (2020/09)



### 0.1. Preprocess NHTS data


### 0.2. Preprocess SafeGraph data


## 1. Execution for ABTS

### 1.0. Data import

###### 데이터 import


### 1.1. Trip Occurence Builder


#### 1.1.0. Data staging
###### 데이터 staging은 한번만 시행해도 되는 것.

##### 1.1.0.1. The probability of a person with age ‘a’ having ‘t’ trips occurs in a single day


##### 1.1.0.2. The number of trips based on unique IDs, age, day type, and trip purpose derived from NHTS data



#### 1.1.1. The number of trips occurring ‘k’ times for a single individual ‘i’ in a day





### 1.2. Trip Chains Builder

#### 1.2.0. Data staging

##### 1.2.0.1. Create trip seqeunce of individuals using origin NHTS data

#### 1.2.1. Finding optimal origin sequence O<i><sub>i</sub></i> most similar to S<i><sub>i</sub></i>
#### 1.2.2. Randomly assign the trip sequence for S<i><sub>i</sub></i>
#### 1.2.3. Reassign the sequence of trip in S<i><sub>i</sub></i> based on O<i><sub>i</sub></i>

###### 여기에는 하나의 equation



### 1.3. Trip Timing Estimator

#### 1.3.0. Data staging

##### 1.3.0.1. Extracting dwell time by trip purpose using NHTS


##### 1.3.0.2. Extracting trip start time by trip purpose using NHTS


#### 1.3.1. Estimating dwell time
#### 1.3.2. Estimating trip start time¶

###### 여기에는 하나의 equation


### 1.4. Trip Mode Assigner



### 1.5. Spatial Trip Route Estimator

#### 1.5.0. Data staging

##### 1.5.0.1. Ratio between straight path and network path


#### 1.5.1. Estimate probabilistic destinations for trip purpose t


#### 1.5.2. Compute trip distance and duration



#### 1.5.3. Optimize Trips with Logical and space-time constraints





## 2. Results (example)