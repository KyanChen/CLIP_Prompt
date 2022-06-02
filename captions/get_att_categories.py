import json

from textblob import TextBlob
from textblob.taggers import NLTKTagger
from textblob.np_extractors import ConllExtractor
import multiprocessing

extractor = ConllExtractor()
nltk_tagger = NLTKTagger()

batch_size = 4

def get_att_categories(data, atts, categories, pid):
    n_data = len(data)
    n_batch = n_data // batch_size
    if n_batch % batch_size != 0:
        n_batch += 1
    for i_batch in range(n_batch):
        text_list = data[i_batch*batch_size: (i_batch+1)*batch_size]
        if i_batch == n_batch - 1:
            text_list = data[i_batch*batch_size:]
    blob = TextBlob(text_list, pos_tagger=nltk_tagger, np_extractor=extractor)
    print(blob.pos_tags)
    print(blob.noun_phrases)
    for word in blob.words:
        print(word.lemmatize())
    atts.append(1)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    n_process = 1

    json_data = json.load(open('', 'r'))['captions']
    data_slice_list = []
    n_item_per_slice = len(json_data) // n_process
    for i in range(n_process):
        start = i * n_item_per_slice
        end = min(start + n_item_per_slice, len(json_data))
        data_slice_list.append(json_data[start: end])

    process_list = []
    atts = multiprocessing.Manager().list()
    categories = multiprocessing.Manager().list()

    for pid in range(n_process):
        slice_datas = data_slice_list[pid]
        print('pid {}'.format(pid))
        process_list.append(
            multiprocessing.Process(target=get_att_categories, args=(slice_datas, atts, categories, pid))
        )
    [p.start() for p in process_list]
    [p.join() for p in process_list]

    atts = list(atts)
    categories = list(categories)
    json_data = {'atts': atts, 'categories': categories}
    json.dump(json_data, open(f'captions_all/extracted_atts_categories.json', 'w'), indent=4)
