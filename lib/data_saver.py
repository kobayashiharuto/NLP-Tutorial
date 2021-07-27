from sklearn.model_selection import train_test_split
import gensim.downloader as api
from tqdm import tqdm
from gensim.models import KeyedVectors


# word2vecの辞書データをダウンロード
wv = KeyedVectors.load_word2vec_format(
    'C:/Users/owner/Desktop/NLP/DATA/fastText/cc.en.300.vec')
wv.save_word2vec_format('cc.en.300.bin', binary=True)
