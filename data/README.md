# 데이터 설명

### delivery.json
- '배달주문' 모델을 만들 때 사용한 데이터.
- 경로 : https://huggingface.co/datasets/mogoi/delivery_all

### reservation.json
- '식당예약' 모델을 만들 때 사용한 데이터.
- 경로 : https://huggingface.co/datasets/mogoi/reservation_all

### refune.json
- '환불' 모델을 만들 때 사용한 데이터.
- 경로 : https://huggingface.co/datasets/mogoi/refund_all

****
## 데이터만들기_re
- 데이터를 어떻게 만들어갔는지의 과정을 보여주기 위한 폴더.
   
#### 폴더
- base : 제일 기본 문장 데이터. 이 데이터를 기반으로 조합 및 placeholder 채우는 등의 과정을 진행한다.
- base_csv : 엑셀 파일을 csv로 변환
- middle : base 폴더 안에 있는 파일을 가지고 만든 데이터. 컬럼명의 숫자가 동일한 데이터을 모든 조합이 나올 수 있도록 조합했다.
- using : middle 데이터 내에 있는 placeholder의 값들을 채워넣는다.
- using_fn : using 폴더 안에 있는 파일들을 concat으로 2~3개의 파일을 묶어줬다.
- z_model_data : 허깅페이스에 올리기 전 데이터를 처리해 json으로 만들었다.

#### 파일
- 찐데전 : 위의 과정을 처리한 파일.
- date : 날짜가 저장된 파일.
- time : 시간이 저장된 파일.
- 세부내용 : placeholder 부분을 채워넣을 데이터를 정리한 파일.
- 예약_상황설정 : 연습 시작 전 어떤 상황인지 알려 최대한 주제에서 벗어나는 말을 하지 않도록 하기 위한 문장 데이터 파일.
