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
- [x] ~~`ray` 사용해서 병렬처리~~
- [ ] `ing` hyper-parameter 튜닝

## Directory tree

- `📄 initial.py` :  **각종 hyper parameter** 설정 해둔 파일
- `📄 Graph.py   ` :  **Graph Model** 파일 (NG, FSCIL방식으로 vertex update 수행. base graph 만들고, 같은 방식으로 incremental learnnig 수행. Edge는 학습에 이용되지 않아 구현하지 않음)
- `📄 run_example.py` : 실행 파일
- `📄 env.yaml` : conda environment export 파일



## Environment 

- import environment : `conda env create -n [env_name] -f env.yaml`

- activate environment : `conda activate [env_name]`



## Accuracy Test - (alpha, eta setting)

### FSCIL adaptation

$$
\Delta{\mathbf{m}}_{r_i} = \eta\cdot e^{-i/\alpha}(\mathbf{f}-\mathbf{m}_{r_i}), i=1\dots N
$$

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



### Neural Gas adaptation

$$
\Delta w_i =\epsilon(t)\cdot h_\lambda(k_i(\xi,\mathcal{A}))\cdot(\xi-w_i),\\
\lambda(t)=\lambda_i(\lambda_f/\lambda_i)^{t/t_{max}},\\
\epsilon(t)=\epsilon_i(\epsilon_f/\epsilon_i)^{t/t_{max}},\\
h_\lambda(k)=exp(-k/\lambda(t))
$$

|   #    | feature # | vertex # | lambda i | lambda f | epsilon i | epsilon f | max_iter |   Train    |
| :----: | :-------: | :------: | :------: | :------: | :-------: | :-------: | :------: | :--------: |
|   1    |   30000   |   1200   |    10    |   0.01   |    0.1    |   0.005   |   1000   |   56.99%   |
|   2    |   30000   |   1200   |    1     |   0.01   |    0.1    |   0.005   |   1000   |   52.58%   |
|   3    |   30000   |   1200   |   0.1    |   0.01   |    0.1    |   0.005   |   1000   |   51.67%   |
|   4    |   30000   |   1200   |   0.01   |   0.01   |    0.1    |   0.005   |   1000   |   51.65%   |
|   5    |   30000   |   1200   |    10    |  0.001   |    0.1    |   0.005   |   1000   |   56.07%   |
|   6    |   30000   |   1200   |    1     |  0.001   |    0.1    |   0.005   |   1000   |   50.61%   |
|   7    |   30000   |   1200   |   0.1    |  0.001   |    0.1    |   0.005   |   1000   |   49.72%   |
|   8    |   30000   |   1200   |   0.01   |  0.001   |    0.1    |   0.005   |   1000   |   50.55%   |
|   9    |   30000   |   1200   |  0.001   |  0.001   |    0.1    |   0.005   |   1000   |   51.22%   |
|   10   |   30000   |   1200   |    10    |   0.01   |     1     |   0.005   |   1000   |   57.36%   |
|   11   |   30000   |   1200   |    10    |   0.01   |    10     |   0.005   |   1000   |   58.62%   |
|   12   |   30000   |   1200   |    10    |   0.01   |    100    |   0.005   |   1000   |   58.39%   |
|   13   |   30000   |   1200   |    10    |   0.01   |   0.01    |   0.005   |   1000   |   58.87%   |
|   14   |   30000   |   1200   |    10    |   0.01   |    0.1    |   0.01    |   1000   |   57.56%   |
|   15   |   30000   |   1200   |    10    |   0.01   |    50     |   0.005   |   1000   |   58.68%   |
|   16   |   30000   |   1200   |   100    |   0.01   |   0.01    |   0.005   |   1000   |   59.15%   |
|   17   |   30000   |   1200   |   1000   |   0.01   |   0.01    |   0.005   |   1000   |   55.40%   |
|   18   |  100000   |   1200   |    10    |   0.1    |    10     |    0.1    |   1000   |   64.97%   |
|   19   |  100000   |   1200   |    10    |   0.1    |    10     |   0.01    |   1000   |   66.29%   |
|   20   |  100000   |   1200   |    10    |   0.1    |    10     |   0.005   |   1000   |   66.41%   |
|   21   |  100000   |   1200   |    50    |   0.05   |     1     |    0.1    |   1000   |   66.16%   |
|   22   |  100000   |   1200   |    50    |   0.05   |     1     |   0.01    |   1000   |   65.32%   |
|   23   |  100000   |   1200   |    50    |   0.05   |     1     |   0.005   |   1000   |   65.34%   |
|   24   |  100000   |   1200   |   100    |   0.01   |   0.01    |   0.005   |   1000   |   66.12%   |
|   25   |  100000   |   1200   |    1     |  0.001   |    0.1    |   0.005   |   1000   |   65.31%   |
|   26   |  100000   |   1200   |   0.1    |  0.001   |    0.1    |   0.005   |   1000   |   61.97%   |
|   27   |  100000   |   1200   |   0.01   |  0.001   |    0.1    |   0.005   |   1000   |   62.70%   |
|   28   |  100000   |   1200   |    1     |   0.1    |    0.1    |   0.005   |   1000   |   64.21%   |
|   29   |  200000   |   1200   |   100    |  0.005   |    0.1    |   0.05    |   1200   |   67.83%   |
|   30   |  200000   |   1200   |   100    |   0.01   |    0.1    |   0.001   |   1200   |   67.98%   |
|   31   |  100000   |   1200   |    50    |   0.01   |    10     |    0.1    |   1200   |   65.69%   |
|   32   |  100000   |   1200   |    50    |   0.01   |    100    |    0.1    |   1200   |   65.76%   |
|   33   |  100000   |   1200   |    50    |   0.01   |     1     |   0.01    |   1200   |   66.11%   |
|   34   |  100000   |   1200   |    50    |   0.01   |    10     |   0.01    |   1200   |   65.86%   |
| **35** |  200000   |   1200   |    10    |   0.01   |    10     |   0.01    |   1200   |   68.60%   |
|   36   |  200000   |   1200   |    50    |   0.01   |    10     |   0.01    |   1200   |   68.16%   |
|   37   |  200000   |   1200   |   100    |   0.01   |    10     |   0.01    |   1200   |   68.21%   |
|   38   |  200000   |   1200   |    10    |   0.01   |     1     |   0.01    |   1200   | **68.81%** |
|   39   |  200000   |   1200   |    50    |   0.01   |     1     |   0.01    |   1200   |   68.41%   |
|   40   |  200000   |   1200   |   100    |   0.01   |     1     |   0.01    |   1200   |   67.73%   |
|   41   |           |          |          |          |           |           |          |            |
|   42   |           |          |          |          |           |           |          |            |
|   43   |           |          |          |          |           |           |          |            |
|   44   |           |          |          |          |           |           |          |            |

