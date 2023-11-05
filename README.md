# 23-2_DSL_Modeling_each_team_project_name
---
## 주제
- 한글 지원이 되지 않는 영문 폰트에 대응하는 한글 폰트 생성
## Team F 김영현 유선재 이성균 장현빈 정성오
---
# Overview
[발표자료](F조_발표자료.pdf)
## 1. Overall Pipeline
![image](https://github.com/younghkim1/23-2_DSL_Modeling_FontGenerator/assets/121621498/4f208206-3f75-4b46-8a4e-e9dc54b33c31)
- 생성하고자 하는 글자의 모양을 가진 source 이미지로부터 글자의 구조와 모양에 대한 정보를 담은 content feature를 뽑기 위해 UNet의 encoder part를 통과시킴.

- UNet의 인코더의 6개의 층으로부터 각각의 레벨에서 structure feature을 뽑아내고, 이들이 잘뽑혔는지 확인하기 위해 다시 UNet의 decoder를 통해 복원하여 확인함.
  
- 원하는 스타일을 가진 target 이미지의 경우 폰트의 스타일에 대한 정보를 가진 feature를 추출하기 위하여 UNet과 별개의 encoder를 통해 style feature를 뽑아냈음.
  
- style feature을 6단계의 de-convolution block을 가진 generator에 입력으로 넣어주는데, 6개의 레벨마다 이전에 UNet encoder에 source image를 통과시켜 6단계에 걸쳐 얻어낸 style features를 condition으로 넣어줌.
  
- 이러한 방식으로 source 이미지와 target 이미지를 결합하여 generator의 최종 아웃풋으로는 32by32 이미지를 얻게 됨.
## 2. Model
![image](https://github.com/younghkim1/23-2_DSL_Modeling_FontGenerator/assets/121621498/aa304b44-70a5-40b5-9931-81b409dd2060)
- 생성된 이미지는 target font의 스타일을 가진 글자이므로 모델이 잘 학습 되었다면 target font에서 가져온 같은 글자 모양 이미지와 같아야 하기 때문에 그 둘 간에 l1 norm의 형태로 generator loss를 정의하였음.
  
- fake image와 source image는 gan의 분포추론을 이용하기 위해 discriminator를 사용해 discriminator loss를 정의하였음.
## 3. Dataset
![image](https://github.com/younghkim1/23-2_DSL_Modeling_FontGenerator/assets/121621498/718f2256-0df0-4756-a9a7-db34122038c8)
- 한글과 영어가 모두 지원이 되는 폰트의 소스파일인 ttf 파일로부터 각 폰트 당 영어 52개와 한글 조합 약 만 천 개의 이미지를 변환하는 방법을 이용하여 학습을 진행함.
# Result
## 1. Final Output
- target font
  
![image](https://github.com/younghkim1/23-2_DSL_Modeling_FontGenerator/assets/121621498/a88dca35-023f-490a-bacb-7a52186aaa9e)
- generated font

![image](https://github.com/younghkim1/23-2_DSL_Modeling_FontGenerator/assets/121621498/040b512d-e135-4d42-951d-6a5a5c55fcde)
## 2. Meaning & Limitations
- 영어 폰트의 경우 52개의 알파벳에 대한 글씨체를 구성하면 끝이지만 한글 폰트의 경우 약 만 천개의 글씨에 대해 모두 글씨체를 구성해야하므로 비용과 시간이 상당히 많이 필요함.
- FontGenerator는 영어만 가능한 폰트가 한글에 대해서도 호환이 가능할 수 있도록 모든 한글 조합에 대하여 이미지를 생성함으로써 폰트를 생성하는데 필요한 시간과 비용을 크게 단축할 수 있게 도와준다는 점에 큰 의미를 가짐.
- 그러나 ttf파일로부터 이미지를 불러오는 속도가 다소 느려서 단 22개의 폰트를 불러오고 모델을 학습하는데 상당히 많은 시간이 소요되었음.
- 모델의 속도를 높이고자 불러온 이미지의 해상도를 32 by 32로 낮춰서 모든 프로세스를 진행하였기 때문에 생성된 폰트의 해상도가 낮음.
- ttf파일을 hdf5라는 형식의 파일로 바꾼 후, 거기에서 파일을 불러와서 학습과 추론을 진행하는 방법을 이용한다면 학습 시간을 크게 단축시킬 수 있을 것으로 생각됨.
- 낮은 해상도로 인한 한계점을 개선하기 위하여 생성된 이미지의 해상도를 높이는 super-resolution 모델을 사용해볼 수 있을 것이라고 생각됨.
# File description
- main (실제 구동하는 파일)
  - ```main.py```  
- model (모델 내부 구조 파일)
  - ```model.py```
- dataset & utils (데이터 생성 파일)
  - ```dataset.py```
  - ```utils.py```
