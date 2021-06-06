import gzip
import json
import math
import multiprocessing
import re
import threading


def convert_json_to_text_file() -> None:
    json_data_filename = '../phase-1/output/fdalabel-query-111031/json_data.json'
    description_filename = 'output/all-description.txt'

    # load labeling from JSON
    with open(json_data_filename, mode='r', encoding='utf-8') as f:
        filenames = json.load(f)

    # extract labels from JSON
    sentences = []
    for filename, labeling in filenames.items():
        sentence = ' '.join(labeling).replace('\\n', ' ').replace('\\t', ' ')
        sentences.append(sentence)

    # replace multiple spaces with single space
    sentences = re.sub('\\s+', ' ', ' '.join(sentences))

    # write description to file
    with open(description_filename, mode='w', encoding='utf-8') as f:
        f.write(sentences)


def read_file_content() -> str:
    description_filename = 'output/all-description.txt'
    with open(description_filename, mode='r', encoding='utf-8') as f:
        return f.read()


def convert_text_to_token(txt: str) -> list:
    return txt.split(' ')


def read_mesh_dict() -> list:
    mesh_dict_filename = 'mesh.json.gz'
    with gzip.open(mesh_dict_filename, 'rb') as f:
        mesh_dict = json.load(f)

    return sorted(mesh_dict.items(), key=lambda t: t[1], reverse=True)


def map_mesh_terms_on_text(mesh_terms: list, sentences: list, thread_counter: int):
    description_bio_schema_filename = f'output/all-description-bio-schema_{thread_counter}.tsv'

    sentence_count = len(sentences)

    with open(description_bio_schema_filename, mode='w', encoding='utf-8') as f:
        for i, sentence in enumerate(sentences):
            print(f'parsing file: {thread_counter}, sentence: {i + 1} of {sentence_count}')

            while not (sentence is None):
                term_found = False

                for mesh_term_key, mesh_term_value in mesh_terms:
                    if sentence.lower().startswith(mesh_term_value.lower()):
                        mesh_term_value_tokens = mesh_term_value.split(' ')
                        sentence_arr = sentence.split(' ')
                        mesh_term_found_in_sentence = ' '.join(sentence_arr[:len(mesh_term_value_tokens)])

                        for j, mesh_term_value_token in enumerate(mesh_term_value_tokens):
                            if j == 0:
                                f.write(f'{sentence_arr[j]}\tB-{mesh_term_key}\n')
                            else:
                                f.write(f'{sentence_arr[j]}\tI-{mesh_term_key}\n')

                        pieces = sentence.split(mesh_term_found_in_sentence, 1)

                        term_found = True
                        # print(f'mesh_term_value: {mesh_term_found_in_sentence}')
                        break

                if not term_found:
                    pieces = sentence.split(' ', 1)
                    others = pieces[0]
                    f.write(f'{others}\tO\n')
                    # print(f'others: {others}')

                # update txt with remaining content and continue search for MESH terms
                sentence = pieces[1].lstrip() if len(pieces) > 1 else None
                # print(f'remaining: {sentence}')

            # write empty line to file after each sentence
            f.write('\n')


def run_thread(mesh_terms: list, txt: str, number_of_thread: int):
    sentences = txt.split('. ')
    sentence_count = len(sentences)

    threads = []

    for i in range(number_of_thread):
        start_index = math.ceil(i * sentence_count / number_of_thread)
        end_index = math.ceil((i + 1) * sentence_count / number_of_thread)
        # print(start_index, end_index)

        selected_sentences = sentences[start_index:end_index]
        x = threading.Thread(target=map_mesh_terms_on_text, args=(mesh_terms, selected_sentences, i + 1))
        threads.append(x)
        x.start()

    for i, thread in enumerate(threads):
        thread.join()


if __name__ == '__main__':
    convert_json_to_text_file()
    text = read_file_content()
    # tokens = convert_text_to_token(text)
    mesh_pairs = read_mesh_dict()

    # configure number of threads
    cpu_count = multiprocessing.cpu_count()
    run_thread(mesh_pairs, text, number_of_thread=cpu_count)
