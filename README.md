# ABTS
Agent Based Travel Scheduler library to predict individual travel schedules


## License
ABTS / version 0.0.3
- install:

```python
!pip install ABTS
```

###### 이 알고리즘은 ~

###### Figure1_Framework_of_ABTS.png 넣기. 그리고 간단히 설명하는데 Classification_method_in_ABTS.png하고 Data_description.png넣기

###### area는 Milwaukee이고 2020년 9월을 예로 하였음.




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