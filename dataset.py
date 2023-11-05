import torch
import numpy as np


class GANDataset(torch.utils.data.Dataset):
    def __init__(self, font_data, UNet_result_feat):
        # font_data: (32*폰트수, 359168) 크기의 폰트 데이터
        #           ex. 폰트가 2개면 font_data의 크기는 (64, 359168)이다.
        # UNet_result_feat: extracted features of each word from UNet model
        self.font_data = font_data
        self.UNet_result_feat = UNet_result_feat

        # 현재는 11224개의 글자가 반영되어 있음!
        # 그래서 지금은 self.total_word_num = 11224인 상태
        self.total_word_num = (font_data).shape[1]//32


    def __len__(self):
        # (폰트의 수*폰트별 글자수)를 반환
        # ex.폰트 2개, 폰트별 글자수 100개면 -> 200을 반환.
        return int(self.target_fonts.shape[0]/32 * self.target_fonts.shape[1]/32)

# 테스터에 사용자 입력: 2개 (index font, index 한글)
# 글씨체 인풋은 22개-> 0 ~ 22 중에 숫자를 받아서
# (숫자*11224)+11172 ~ (숫자*11224)+11224: A-z(52개) -> generator & 문자 생성 -> 평균 -> 출력
# ex.
# 궁처체: 0~11223
# 신명조: 11224 ~ 22447
# ...


    def __getitem__(self, font_idx, font_eng_idx, word_idx):
        # font_idx: 원하는 폰트의 idx. 현재는 폰트 수가 22개 이므로 0 ~ 21까지 입력 가능
        # font_eng_idx: 원하는 폰트에서 영어의 index. A~z까지 총 52개 이므로 0~51입력 가능
        #           단어를 생성할 때 사용자가 입력한 한글과 A~z까지 모든 pair에 대해
        #           이미지를 생성해 평균내서 최종 이미지를 만들 것이므로 이때 font_eng_idx를 사용함.
        # word_idx: 생성하고 싶은 글자의 idx.
        #           현재 폰트별로 한글이 11172개 있으므로 0~11171까지 입력 가능


        # 지금 generator 모델의 input은 총 3개: ex)
        # 1.궁서체 a, 2.사용자가 원하는 글자, 3.해당 글자의 feature(from UNet)


        # index 0~11171까지는 한글, index 11171~11224까지는 영어
        # idx라는 인풋은 특정 폰트에 해당하는 특정 글자의 index이다. ex) index = 0: 가(궁서)
        font = font_idx # 글자 하나하나를 독립적으로 봄. ex) 0:가(궁서), 4053:가(신명조)
        word = word_idx

        # 해당 폰트의 A~z 중에서 font_eng_idx에 해당하는 영어글자를 선택 (52개의 글자를 선택)
        # 현재 359168 = (11,224*32) 즉, 11224개의 글자가 각각의 32개의 원소를 가지고 있는 형태
        # 따라서 인덱스 0 ~ 357503이 한글을 나타내고
        # 359168-52*32 = 357504부터 영어 A가 시작된다.
        # 글씨의 해상도가 높아지면 32를 해당 해상도로 변경해줘야 함.
        eng_start_idx = self.total_word_num*32 - 52*32
        eng = font_eng_idx
        eng_word = self.font_data[font*32:(font+1)*32,
                                  eng_start_idx + eng*32 : eng_start_idx + (eng+1)*32]

        # 해당 폰트에서 사용자가 원하는 한글 글자(word)를 선택
        korean_word = self.font_data[font*32:(font+1)*32, word*32:(word+1)*32]

        # 현빈이가 만든 원본
        # return {"eng_word":eng_word,
        #         "korean_word":korean_word,
        #         "word":word,
        #         # UNet에서 뽑은 feature: 제일 처음 기본 폰트 4052개에 대한 feature (6*4052)
        #         "emb":(self.UNet_result_feat['cl1'][word],
        #                 self.UNet_result_feat['cl2'][word],
        #                 self.UNet_result_feat['cl3'][word],
        #                 self.UNet_result_feat['cl4'][word],
        #                 self.UNet_result_feat['cl5'][word],
        #                 self.UNet_result_feat['cl6'][word])}

        return {"eng_word":eng_word,
        "korean_word":korean_word,
        # 현재 상황은 테스트를 하는 상황이기 때문에 해당 폰트에 해당하는
        # korean_word는 딱히 필요한 데이터는 아님

        "word":word,
        # UNet에서 뽑은 feature: 제일 처음 기본 폰트 4052개에 대한 feature (6*4052)
        "emb":(self.UNet_result_feat['cl1'][word].reshape((1,) + self.UNet_result_feat['cl1'][word].shape),
                self.UNet_result_feat['cl2'][word].reshape((1,) + self.UNet_result_feat['cl2'][word].shape),
                self.UNet_result_feat['cl3'][word].reshape((1,) + self.UNet_result_feat['cl3'][word].shape),
                self.UNet_result_feat['cl4'][word].reshape((1,) + self.UNet_result_feat['cl4'][word].shape),
                self.UNet_result_feat['cl5'][word].reshape((1,) + self.UNet_result_feat['cl5'][word].shape),
                self.UNet_result_feat['cl6'][word].reshape((1,) + self.UNet_result_feat['cl6'][word].shape))}


    def __repr__(self):
        return f"data size : {self.__len__()} "


# load font file which is font.npz and make it to a Dataset
def load_and_make_font_dataset(UNet_result_feat, font_path = "."):
    # font_path: path of font.npz
    # ex. "/content/drive/MyDrive/FancyFont/현빈"
    datasets = np.load(font_path+"/font.npz")
    source_fonts = datasets['source_fonts']
    font_data = datasets['target_fonts'] # 제대로 쓰는 건 이 font_data만 쓴다.

    font_dataset = GANDataset(font_data, UNet_result_feat)
    return font_dataset
