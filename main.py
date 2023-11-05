import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from model import *
from dataset import *
from utils import *


def main():
    # UNet_result_feat.npz 파일이 있는 경로 설정
    # ex."/content/drive/MyDrive/FancyFont/현빈"
    UNet_result_feat_path = ""


    UNet_result_feat = load_UNet_result_feat(UNet_result_feat_path)



    # generator_result.pt 파일이 있는 경로 설정
    # ex."/content/drive/MyDrive/FancyFont/현빈"
    generator_result_path = ""


    generator = load_trained_model(generator_result_path)

    device = torch.device("cuda") if torch.cuda.is_available()\
            else torch.device("cpu")
    print(f"Current Device: {device}")

    generator = generator.to(device)



    # font.npz 파일이 있는 경로 설정
    # ex. "/content/drive/MyDrive/FancyFont/현빈"
    font_path = ""



    font_dataset = load_and_make_font_dataset(UNet_result_feat, font_path=font_path)

    font_idx, input_sentence = enter_font_and_sentence()
    word_idx_list = kor_to_idx(input_sentence)
    generated_img_list = generate_images(generator, font_dataset,
                                        font_idx, word_idx_list)
    show_generated_images(generated_img_list)



    # path for saving the generated images
    saving_path = ''


    # save the generated image at the "saving_path"
    # saved file name: 'font'+str(font_idx)+'_'+input_sentence+'.npy
    #                   ex. 'font3_안녕하세요.npy'
    # file format: ndarray with
    #      "# of generated letters" * height(=32) * width(=32) * channel(=1) size
    #      ex. input sentence: 안녕하세요 -> 5*32*32*1 size ndarray
    save_img_as_npy(font_idx, input_sentence, generated_img_list, saving_path)


if __name__ == '__main__':
    main()