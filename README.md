# Stroke diagnosis
LinkBERT, Bi-GRU 등으로 이중언어로 된 뇌졸중 판독문을 학습시켜 뇌졸중 여부를 판단.   
<br/>
model weight download
> https://www.notion.so/Stroke-AI-Contest-13e5bc17aaac4b82a1b80465fff34684
<br/>

* Dataset<br/>
<br/>
|Findings|Conclusion|Acutelnfarction|
|--------|----------|---------------|
|문장    |문장       |Label          |
위와 같은 form으로 된 csv format data<br/>
문장은 영어와 한글이 혼용된 이중언어 문장임.<br/>
<br/>
* task

data를 input으로 넣으면 뇌졸중 여부를 0, 1로 classification.
