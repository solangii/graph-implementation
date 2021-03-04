# Graph Implementation

Meta Continual learning을 위한 graph implemetation repository

- FSCIL Repository : https://github.com/xyutao/fscil
- Neural Gas Paper : http://ftp.ks.uiuc.edu/Publications/Papers/PDF/MART91B/MART91B.pdf
- Graph 작동 방식 이해하기 : https://www.notion.so/Graph-1ef65d4d6f084b7380345a8ce0f5053b



## To do

- [x] `210216 done` z, c, lambda error 
- [x] `210216 done` GPU 연산으로 바꾸기  
- [x] `210216 done` Accuracy 측정 function 만들기 (for hyperparameter-tuning)
- [x] `210217 done` m update for incremental phase 일단 식3 방식으로 implement 해보기
- [x] `210225 done`  feature set load file 
- [x] `210225 done` accuracy test error 해결
- [ ] `ing` `ray` 사용해서 병렬처리
- [ ] `ing` hyper-parameter 튜닝

## Directory tree

- `📄 initial.py` :  **각종 hyper parameter** 설정 해둔 파일
- `📄 Graph.py   ` :  **Graph Model** 파일 (NG, FSCIL방식으로 vertex update 수행. base graph 만들고, 같은 방식으로 incremental learnnig 수행. Edge는 학습에 이용되지 않아 구현하지 않음)
- `📄 run_example.py` : 실행 파일



## Accuracy Test

|  #   | feature # | vertex # | alpha |    eta    | max_iter |   Train    |  Test  |
| :--: | :-------: | :------: | :---: | :-------: | :------: | :--------: | :----: |
|  1   |   5000    |   200    |  10   |    0.1    |   1000   |    26%     |        |
|  2   |   10000   |   400    |  10   |    0.1    |   1000   |   38.49%   |        |
|  3   |   20000   |   800    |  10   |    0.1    |   1000   |   42.85%   |        |
|  4   | **30000** | **1200** |  10   |    0.1    |   1000   |   54.76%   |        |
|  5   |  100000   |   1200   |  10   |    0.1    |   1000   |   53.79%   | 53.23% |
|  6   |   30000   |   1200   |  30   |    0.1    |   1000   |   34.96%   | 32.27% |
|  7   |   30000   |   1200   |  20   |    0.1    |   1000   |   44.14%   | 42.53% |
|  8   |   30000   |   1200   |   5   |    0.1    |   1000   |   59.25%   | 58.13% |
|  9   |   30000   |   1200   |   3   |    0.1    |   1000   |   60.86%   | 57.9%  |
|  10  |   30000   |   1200   | **1** |    0.1    |   1000   | **61.62%** | 58.67% |
|  11  |   30000   |   1200   |   5   |   0.01    |   1000   |   44.79%   | 41.27% |
|  12  |   30000   |   1200   |   5   |   0.05    |   1000   |   59.96%   | 55.53% |
|  13  |   30000   |   1200   |   5   | **0.075** |   1000   |   60.36%   | 57.63% |
|  14  |   30000   |   1200   |   5   |     1     |   1000   |   35.11%   | 33.43% |
|  15  |   30000   |   1200   | **1** | **0.075** |   1000   |   59.5%    | 55.58% |