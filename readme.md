# Graph Implementation

Meta Continual learning을 위한 graph implemetation repository

- FSCIL Repository : https://github.com/xyutao/fscil
- Neural Gas Paper : http://ftp.ks.uiuc.edu/Publications/Papers/PDF/MART91B/MART91B.pdf
- Graph 작동 방식 이해하기 : https://www.notion.so/Graph-1ef65d4d6f084b7380345a8ce0f5053b



## To do

- [x] `210216 done` z, c, lambda error 
- [x] `210216 done` GPU 연산으로 바꾸기  
- [x] `210216 done` Accuracy 측정 function 만들기 (for hyperparameter-tuning)
- [x] ~~Base Graph Module화~~ -> 어차피 객체라서 굳이 안해도 될 것 같음
- [x] `210217 done` m update for incremental phase 일단 식3 방식으로 implement 해보기
- [x] ~~`ing` gpu number 지정 옵션추가 시도해보기~~ ->어차피 메인코드에서 해결 ㄱㄴ
- [ ] `ing` @hmcoo // feature set load file ~~(화이팅)~~
- [ ] `tmrw` 함수별 description 작성하기

feature set load 완료되면 hyper-parameter tuning이랑 적절하게 accuracy 나오는지 확인 거치면 graph phase끝납니다! (eq3도 변경해야하지만..)



## Directory tree

- `📄 initial.py` :  **각종 hyper parameter** 설정 해둔 파일
- `📄 Graph.py   ` :  **Graph Model** 파일 (현재까지는 Neural Gas방식으로 base graph만 만듬)
- `📄 run_example.py` : 실행 파일

