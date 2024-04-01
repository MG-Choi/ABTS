# ABTS
Agent Based Travel Scheduler library to predict individual travel schedules


## License
ABTS / version 0.0.3
- install:

```python
!pip install ABTS
```

###### 이 알고리즘은 개개인의 travel schedule을 예측하는 시뮬레이션으로, ABTS (Agent-Based Travel Scheduler)이다. 파라미터 4가지를 이용하여 예측을 하며, 기존의 OD데이터를 이용하여 과거의 travel을 trip purpose, age 별로 decompose할 수 있고, 미래 예측 역시 가능하다. 자세한 사항은 다음의 두 논문에서 찾을 수 있다.

- Choi, M., & Hohl. A. (2024). Derivation of Spatiotemporal Risk Areas and Travel Behaviors During Pandemic Through Reverse Estimation of Mobility Patterns by Agent-Based Modeling, Spatial and Spatio-temporal Epidemiology. Under review (1st round)

- Choi, M., Seo, J., & Hohl. A. (2024). ABTS: Agent-Based Travel Scheduler. In process


###### Figure 1 shows the framework of ABTS. ABTS utilizes a three-step process to generate the initial outcome of individual travel schedules: 1) Comprehensive Travel Classifier, 2) Land-Use Estimator, and 3) Individual Travel Schedule. The output of ABTS includes individual daily trip chains that consists of each trip purpose, occurrence time, destination, duration, and mode, tailored according to age group.

![Figure1. Framework_of_ABTS](/ABTS/image/Figure1_Framework_of_ABTS.png)

###### Comprehensive travel classifier에서는 Table 1과 같이 trip purpose와 trip mode, Age group을 각각의 기준으로 classify한다. 예로 trip purpose는 9개의 Major trip purpose와 13개의 Sub trip purpose로 나눠진다.


![Table 1. Classification in ABTS](/ABTS/image/Classification_method_in_ABTS.png)




# Run the model with example data

###### 여기서는 Individual travel schedule generator의 과정을 예제 데이터를 통해 돌려보도록 한다. case area는 Milwaukee이고 2020년 9월의 travel을 예측하는 과정을 보여주겠다.

##### 여기에 Data description

## 0. Preprocessing (using sample simulation in library)

###### 여기 프리프로세싱에는 origin data들 2개 - SafeGraph와 NHTS를 가공하는 단계.

### 0.0. Data import

###### 구글 드라이브에서 데이터 


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