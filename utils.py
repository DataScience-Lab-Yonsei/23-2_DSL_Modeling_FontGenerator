import numpy as np
import random

# 한글을 넣어주면 해당 인덱스를 알려주는 함수

def kor_to_idx(input_sentence):
    co = "0 1 2 3 4 5 6 7 8 9 A B C D E F"
    start_k = "AC00"
    end_k = "D7A3"

    co = co.split(" ")

    korean = [a+b+c+d
                        for a in co
                        for b in co
                        for c in co
                        for d in co]

    korean = np.array(korean)

    s = np.where(start_k == korean)[0][0]
    e = np.where(end_k == korean)[0][0]

    korean = korean[s : e + 1]
    korean_chr = [chr(int(code, 16)) for code in korean]

    start_L = 65
    end_L = 90

    Alphabets = [chr(i) for i in range(start_L, end_L + 1)]

    start_s = 97
    end_s = 122

    alphabets = [chr(j) for j in range(start_s, end_s + 1)]

    english = Alphabets + alphabets

    word_combination = korean_chr + english

    character_indices = []

    for char in input_sentence:
        if char == '':
            # blank는 -1이 들어가도록 설정
            character_indices.append(-1)
        try:
            index = word_combination.index(char)
            character_indices.append(index)
        except ValueError:
            character_indices.append(-1)

    return character_indices


# 받은 글자가 한글인지 확인해주는 함수
def is_only_hangul(s):
    # 한글 유니코드 범위: ㄱ(0xAC00) ~ 힣(0xD7A3)
    for char in s:
        if char == ' ':  # 띄어쓰기 허용
            continue
        if not (0xAC00 <= ord(char) <= 0xD7A3):
            return False
    return True


# 사용자에게 원하는 폰트의 스타일과 글자를 입력받는 함수
# 숫자는 0 ~ 22만 받아야 한다!!
def enter_font_and_sentence():
    while True:
        try:
            user_input = input("원하는 폰트의 스타일을 입력하세요(0 ~ 22): ")
            # 입력값이 정수인지 확인
            if user_input.isdigit() or (user_input.startswith('-') and user_input[1:].isdigit()):
                number = int(user_input)
                if 0<=number<=22: # 0 ~ 22 구간에 들어왔을 때만 인정해줌.
                    break
                else:
                    raise ValueError
            else:
                print("올바른 정수를 입력해주세요!")
        except ValueError:  # 입력이 정수가 아닐 경우 예외 처리
            print("올바른 정수를 입력해주세요!")

    while True:
        user_input = input("글씨체를 변환할 한글을 입력해주세요: ")

        if is_only_hangul(user_input):
            break
        else:
            print("한글만 입력해주세요!")

    return number, user_input


# 입력해준 font로 word_idx에 해당하는 한글 글자를 모델을 통해 만들어서
# 32*32*1 크기의 이미지로 반환해주는 함수
def generate_img(generator, font_dataset, font_idx, word_idx):
    with torch.no_grad():
        generator.eval()
        # 모델에서 각각의 영어 알파벳에 대해 만들어진 이미지를 저장하기 위한 리스트
        generated_img_list = []

        # 영어 알파벳 52개에 대해 각각의 결과를 뽑아서 평균낸다.
        for font_eng_idx in range(52):

            font_d = font_dataset.__getitem__(font_idx, font_eng_idx, word_idx)
            x = font_d['eng_word'].reshape(-1,1,32,32)/255
            x = torch.Tensor(x).to(device)
            catemb = [emb.to(device) for emb in font_d['emb']]

            output = generator(x, *catemb)
            generated_img_list.append(output)

    final_img = sum(generated_img_list)/52
    return final_img.cpu().squeeze(dim=0).permute(1,2,0)


def generate_images(generator, font_dataset, font_idx, word_idx_list):
    num_words = len(word_idx_list)
    generated_img_list = []

    for i in range(num_words):
        if  word_idx_list[i] == -1: # 띄어쓰기의 경우
            generated_img = np.ones((32,32,1)) # 그냥 흰색 이미지를 하나 생성

        else:
            generated_img = generate_img(generator, font_dataset, font_idx,
                                         word_idx_list[i]).numpy()

        generated_img_list.append(generated_img)
    return np.array(generated_img_list, dtype = float)


# 여러개의 이미지를 생성해서 출력
## font_idx: 원하는 폰트 스타일
## word_idx_list: 원하는 단어의 index가 들어있는 리스트
##                          ex) [0,1,11171]:[가, 각, 힣]

def show_generated_images(generated_img_list):
    num_words = len(generated_img_list)

    fig = plt.figure()
    rows = 1
    cols = num_words

    for i, generated_img in enumerate(generated_img_list):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.axis('off')
        if np.all(generated_img==1):
            # inpet sentence에 띄어쓰기가 있어서 흰색 이미지 출력하는 경우
            ax.imshow(generated_img, cmap = "gray", vmin = 0, vmax = 1)
        else:
            ax.imshow(generated_img, cmap = "gray")
    plt.show()


# 영어 폰트와 영어 단어를 입력하면 해당 폰트로 단어를 보여준다.
# 이를 통해 생성된 단어와 기존 영어 폰트를 비교할 수 있다.

## input
## datasets: test를 하는 데 쓰이는 데이터셋
## font_idx: 원하는 폰트의 인덱스
## letters: 확인하고 싶은 영어 알파벳 ex) "apple"
def show_english_font(datasets, font_idx, letters):
    if letters.isalpha() and letters.isascii():
        pass
    else:
        print("영어만 입력하세요.")
        return

    eng_upper_alphabets = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    eng_lower_alphabets = eng_upper_alphabets.lower()
    eng_alphabets = eng_upper_alphabets+eng_lower_alphabets

    alphabet_idx_list = []

    for letter in letters:
        if letter in eng_alphabets:
            alphabet_idx_list.append(eng_alphabets.index(letter))

    word_idx = 0 # it has no meaning
                # just for implementing datasets.__getitem__() function

    num_words = len(alphabet_idx_list)

    fig = plt.figure()
    rows = 1
    cols = num_words

    for i, eng_idx in enumerate(alphabet_idx_list):
        font_d = datasets.__getitem__(font_idx, eng_idx, word_idx)
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(font_d['eng_word'], cmap = "gray")
        ax.axis('off')
    plt.show()


### npy 파일로 저장
# 저장된 파일은 (# of letters, Width, Height, Channel) 크기의 numpy array
# 파일명은 font 1을 사용해 '데싸랩'이란 단어를 만들었을 경우: "font1_데싸랩"

def save_img_as_npy(font_idx, input_sentence, generated_img_list,
                    saving_path='.'):
    img_title = '/font'+str(font_idx)+'_'+input_sentence
    np.save(saving_path + img_title, generated_img_list)


# This function loads saved images in npy file format and shows the images.
## <input>
## img_file_name: name of the saved images in npy format
## saved_path: location of the saved images
## ex.
## img_file_name = 'font1_안녕.npy'
## saved_path = '/content/drive/MyDrive/sungoh/generated_font'

def load_saved_image(img_file_name, saved_path='.'):
    generated_img_list = np.load(saved_path+"/"+img_file_name)
    return generated_img_list


