import glob
import io
import os
import zipfile

import pandas as pd
import requests
from bs4 import BeautifulSoup

from helper.dump_to_json import dump_to_json


def download_files():
    global base_dir

    data = pd.read_csv('fdalabel-query-111031.csv')

    out_dir = 'xml-files/'
    last_set_id = open(base_dir + 'temp_file.txt', 'r').read() if os.path.isfile(base_dir + 'temp_file.txt') else ''
    searching_set_id = True

    for i, SET_ID in enumerate(data['SET ID']):
        # skip all SET_ID up till last_set_id
        if last_set_id != '' and searching_set_id and SET_ID != last_set_id:
            continue
        searching_set_id = False

        # save current SET_ID to temp file
        open(base_dir + 'temp_file.txt', 'w').write(SET_ID)

        # download ZIP file
        link = f'https://dailymed.nlm.nih.gov/dailymed/getFile.cfm?setid={SET_ID}&type=zip'

        try:
            r = requests.get(link)
            print(f'fetching file [{i + 1}]: {r.url}')
            z = zipfile.ZipFile(io.BytesIO(r.content))
        except Exception as e:
            print(e)
            open(base_dir + 'skipped_files.txt', 'a').write(SET_ID + '\n')
            continue

        list_of_file_names = z.namelist()

        for fileName in list_of_file_names:
            # extract XML file ONLY
            if fileName.endswith('.xml'):
                z.extract(fileName, out_dir)

    # empty temp file when all XML files are downloaded
    open(base_dir + 'temp_file.txt', 'w').write('')


def parse_xml(filename):
    global base_dir

    file_content = open(filename, mode='r', encoding='utf-8').read()
    soup = BeautifulSoup(file_content, 'lxml')
    labeling = []

    codes = ['34066-1', '43685-7', '34084-4']
    for code in codes:
        content_items = soup.find(attrs={'code': code})
        if content_items is None:
            continue

        labeling.append(content_items.parent.text.strip())

    return labeling


if __name__ == '__main__':
    base_dir = 'output/fdalabel-query-111031/'

    # download XML files
    download_files()

    # download labeling in JSON format
    xml_files = [f for f in glob.glob('xml-files/*.xml')]
    labels = {}

    for i, xml_file in enumerate(xml_files):
        print(f'parsing file [{i}]: {xml_file}')
        labels[xml_file] = parse_xml(xml_file)

    dump_to_json(labels, base_dir + 'json_data.json', indent=False)
