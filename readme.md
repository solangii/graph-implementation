# Graph Implementation

Meta Continual learning을 위한 graph implemetation repository

- FSCIL Repository : https://github.com/xyutao/fscil
- Neural Gas Paper : http://ftp.ks.uiuc.edu/Publications/Papers/PDF/MART91B/MART91B.pdf
- Graph 작동 방식 이해하기 : https://www.notion.so/Graph-1ef65d4d6f084b7380345a8ce0f5053b



## Todo

- [x] `210216 done` z, c, lambda error 
- [x] `210216 done` GPU 연산으로 바꾸기  
- [ ] Accuracy 측정 function 만들기 (for hyperparameter-tuning)
- [ ] Base Graph Module화
- [ ] @hmcoo // feature set load file 
- [ ] m update for incremental phase 일단 식3 방식으로 implement 해보기
- [ ] gpu number 지정 옵션추가?



## Directory tree

- `📄 initial.py` :  **각종 hyper parameter** 설정 해둔 파일
- `📄 Graph.py   ` :  **Graph Model** 파일 (현재까지는 Neural Gas방식으로 base graph만 만듬)
- `📄 run_example.py` : 실행 파일

