파이썬, 아나콘다를 사용했습니다. ( 사용언어 : python=3.10.6, 컴파일러 : gcc=9.3.0)

먼저 아나콘다 가상환경을 생성 후 진행해주시기 바랍니다.

conda create -n bi-gru python=3.10.6

conda activate bi-gru


---라이브러리 버전---
python==3.10.6
gcc==9.3.0

pandas==1.4.4
numpy==1.23.4
tensorflow==2.10.0
keras==2.10.0
nltk==3.7
scikit-learn==1.1.2

위의 환경을 설치해야 합니다.

혹시 가상환경에서 오류가 난다면 포함되어 있는 bi_gru.yaml을 conda env create -f bi_gru.yaml 해서 conda activate bi_gru 해주시기 바랍니다.

python bi-gru.py > output.txt 명령어를 terminal에 치시면 해당 폴더로 output.txt 파일이 만들어지고, inference 결과가 저장됩니다.
bracket이 없는 조금 더 보기 편한 결과는 폴더에 result.txt로 저장되어 있으니 참고 바랍니다.

기술 문서는 압축 파일 안에 2차 보고서.pdf로 저장되어 있습니다. 

좋은 대회 열어주셔서 감사합니다.