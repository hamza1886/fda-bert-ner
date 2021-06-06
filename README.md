# Train NER model on selected FDA labeling

## TODO

- [x] Use setid in [csv file](phase-1/fdalabel-query-111031.csv) file to download XML files.
- [x] Use [MedDRA Terminology](phase-1/llt.csv) instead of MESH terms, and use exact match to extract MedDRA from labeling section (e.g., boxed warnings, warning and precaution, and adverse reaction).
- [x] Train classifier using [phase-2/output/all-description-bio-schema.tsv](phase-2/output/all-description-bio-schema.tsv) as input to SpaCy/BERT.

## Working

### Phase 1

Download XML files containing MedDRA from labeling section (e.g., boxed warnings, warning and precaution, and adverse reaction).

- Run [phase-1/download_labeling.py](phase-1/download_labeling.py) to download XML files against setid in [csv file](phase-1/fdalabel-query-111031.csv).
- The output is saved in [json_data.json](phase-1/output/fdalabel-query-111031/json_data.json).

### Phase 2

Convert content of [json_data.json](phase-1/output/fdalabel-query-111031/json_data.json) to single long text and write to file [phase-2/output/all-description.txt](phase-2/output/all-description.txt).

- Run [phase-2/convert_text_to_bio_schema.py](phase-2/convert_text_to_bio_schema.py) to convert [phase-2/output/all-description.txt](phase-2/output/all-description.txt) to output BIO schema format.
- The output is saved in
    - [phase-2/output/all-description-bio-schema_1.tsv](phase-2/output/all-description-bio-schema_1.tsv)
    - [phase-2/output/all-description-bio-schema_2.tsv](phase-2/output/all-description-bio-schema_2.tsv)
    - [phase-2/output/all-description-bio-schema_3.tsv](phase-2/output/all-description-bio-schema_3.tsv)
    - [phase-2/output/all-description-bio-schema_4.tsv](phase-2/output/all-description-bio-schema_4.tsv)
- The four output files from previous step are manually combined and saved in [phase-2/output/all-description-bio-schema.tsv](phase-2/output/all-description-bio-schema.tsv).

### Phase 3

Train NER classifier using SpaCy/BERT.

- Manually split [phase-2/output/all-description-bio-schema.tsv](phase-2/output/all-description-bio-schema.tsv) into 60% [train](phase-3/data/train.tsv), 20% [validate](phase-3/data/devel.tsv) and
  20% [test](phase-3/data/train.tsv) data.
- Use [train](phase-3/data/train.tsv) and [devel](phase-3/data/devel.tsv) data as input to train SpaCy/BERT model.
- Use [test](phase-3/data/test.tsv) data to test the trained model.
- Train and test the model
    - Option 1
        - Run [phase-3/train_custom_ner_with_spacy.ipynb](phase-3/train_custom_ner_with_spacy.ipynb) to train the model.
        - Run [phase-3/test_custom_ner_with_spacy.ipynb](phase-3/test_custom_ner_with_spacy.ipynb) to test the model.
    - Option 2
        - Run [phase-3/spacy_ner.py](phase-3/spacy_ner.py) to train and test the model.
