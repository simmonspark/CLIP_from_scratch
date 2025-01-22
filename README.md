# CLIP_from_scratch

내 첫 멀티모달 raw 구현이다.

![스크린샷 2025-01-21 18-20-34](https://github.com/user-attachments/assets/bce540a9-dfb7-4464-a042-eb752d737d34)

![스크린샷 2025-01-21 18-20-47](https://github.com/user-attachments/assets/afa38ade-48d9-4848-967a-a0a5a7468e09)


멀티모달의 기초가 되는 clip을 직접 구현하면서 논문의 그림의 중요성을 알았다. gpt에 의존하지 않고 그림을 분석하고 sudo code를 분석할 때 진짜 실력이 늘어감을 느낀다.

text-img-embedding vector를 contrastive learning을 통해 같은 공간상에 매치시킨다.

결론은 이거다. 이미지의 embedding vector는 해당 text를 설명하는 embedding vector랑 같은 위치에 있게 학습시킨다.

그러면 img query작업을 진행할 수 있다.

text를 query로 쓰고 전체 데이터를 돌면서 crossentrophy loss를 보면서 일정 threshold 이하로 떨어진 이미지만 묶어서 이미지 검색을 할 수 있다.

학습코드가 드디어 좀 성숙해진 것 같다. npy file을 만들지 않고 간단하면서도 IO병목을 해결하는 방식을 사용하고 있다.

학습을 다 완료하고 there is a dog를 query로 img 검색을 해봤다.

![myplot](https://github.com/user-attachments/assets/e222a549-0345-44c0-87a2-e82644fd31db)

이번에는 query를 there is a ship으로 해서 다시 inference를 수행했다.

![ship](https://github.com/user-attachments/assets/af28c484-c118-446a-82b5-ddd5ba5864bc)

각각 query에 맞게 img를 검색한 것을 볼 수 있다.
