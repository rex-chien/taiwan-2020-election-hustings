import os
import fnmatch
from ckiptagger import WS, POS, NER
from wordcloud import WordCloud

ignored_pos = {'V_2', 'DE', 'SHI', 'COLONCATEGORY', 'COMMACATEGORY', 'DASHCATEGORY', 'DOTCATEGORY', 'ETCCATEGORY',
               'EXCLAMATIONCATEGORY', 'PARENTHESISCATEGORY', 'PAUSECATEGORY', 'PERIODCATEGORY', 'QUESTIONCATEGORY',
               'SEMICOLONCATEGORY', 'SPCHANGECATEGORY', 'WHITESPACE',
               'Nh', 'Nf', 'T', 'Nep', 'D', 'Neu', 'Di', 'Caa', 'Cab', 'Cba', 'Cbb'}


def init_ckip_models():
    from ckiptagger import data_utils
    if not os.path.isdir('data'):
        data_utils.download_data('./')

init_ckip_models()


ws_cls = WS('data')
pos_cls = POS('data')
ner_cls = NER('data')
wordcloud = WordCloud(
    background_color='white',
    font_path='C:\\Windows\\Fonts\\msjh.ttc',
    width=800,
    height=600
)


def find_files_in_dir(directory, file_pattern):
    for root, dir_names, file_names in os.walk(directory):
        for file_name in file_names:
            if fnmatch.fnmatch(file_name, file_pattern):
                yield os.path.join(root, file_name)


def main():
    global ws_cls
    global pos_cls
    global ner_cls
    global wordcloud

    src_files = find_files_in_dir('./transcripts/2019-12-29-president-debate', '*.txt')
    for file_path in src_files:
        with open(file_path, encoding='utf-8') as f:
            content = '\n'.join(f.readlines())
        dirname = os.path.dirname(file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Run WS-POS-NER pipeline
        sentence_list = [content]

        word_sentence_list = ws_cls(sentence_list)
        pos_sentence_list = pos_cls(word_sentence_list)
        entity_sentence_list = ner_cls(word_sentence_list, pos_sentence_list)

        terms_freq = {}
        for word, pos in zip(word_sentence_list[0], pos_sentence_list[0]):
            if pos not in ignored_pos:
                count = terms_freq.get(word, 0)
                terms_freq[word] = count + 1

        output_file_path = os.path.join(dirname, f'{file_name}_tf.dat')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            lines = [f'{word},{terms_freq[word]}\n' for word in terms_freq]
            f.writelines(lines)

        output_image_path = os.path.join(dirname, f'{file_name}_tf.png')
        wordcloud.generate_from_frequencies(terms_freq)
        wordcloud.to_file(output_image_path)

        entity_set = set()
        for entity in sorted(entity_sentence_list[0]):
            entity_set.add((entity[3], entity[2]))

        output_file_path = os.path.join(dirname, f'{file_name}_ner.dat')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            lines = [f'{entity[0]},{entity[1]}\n' for entity in entity_set]
            f.writelines(lines)

    # Release model
    del ws_cls
    del pos_cls
    del ner_cls


if __name__ == "__main__":
    main()
