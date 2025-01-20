'''
    Dataset normalisation for DutchMedLM data

    We want to end up with:

        "id": Value("string"), REQUIRED
        "text": Value("string"), REQUIRED
        "source": Value("string"),
        "approx_token_counts_original": Value("int64"),
        "approx_token_counts_translated": Value("int64"),

    if the non-required fields are not present, we will assume default values
    "not available" for "source" and -1 for the token counts
'''
import json
import pandas as pd
import argparse
import os
from google.cloud import storage
import ftfy as ftfy
import tqdm
import re

replacements = [
    (r'\s{2,}', ' ')
]
replacements_re = [(re.compile(repl[0]), repl[1]) for repl in replacements]

meta_data = {
    'MariaNMT_cardiomyopathy.json ': {
                                        'id_field': 'id',
                                        'text_field': 'text'
                                    },
    'PubmedPatients_nllb200.jsonl':{
                                     'id_field':'patient_uid',
                                      'text_field':'text'
                                    },
    'meditron_guidelines_gpt4omini.json': {
                                    'id_field':'id',
                                    'text_field':'text'
    },
    'DeepL_PUBMED_ABSTRACTS_15345931.parquet': {
                                    'id_field':None,
                                    'text_field':'text'
    },
    'Meditron_open_guidelines_translated_MariaNMT.json': {
                                    'id_field': 'id',
                                    'text_field': 'text'
    },
    'PMC-Patients-V2_discharge_dutch_gemini-1.5-flash.jsonl': {
                                    'id_field': 'patient_uid',
                                    'text_field': 'transformed_text'
    },
    'Scraped.jsonl': {
                                    'id_field': 'id',
                                    'text_field': 'text'
    },
    'apollo_guidelines_MariaNMT.json': {
                                    'id_field': 'id',
                                    'text_field': 'text'
    },
    'apollo_guidelines_nllb200.json': {
                                    'id_field': 'id',
                                    'text_field': 'text'
    },
    'apollo_medicalGuideline_en_text_GPT4o_mini.json':{
                                    'id_field': 'id',
                                    'text_field': 'text'
    },
    'GPT4omini_pubmed_cardiomyopathy.json': {
                                    'id_field': 'id',
                                    'text_field': 'text'
    }
}

# things to add: repetitions of non-word characters, punctuation, whitespace and linebreaks
# things to add: repetitions of words

def clean_text(text):
    # fix encoding issues
    text = ftfy.fix_text(text)
    text = text.strip()

    # repeated non-word characters
    text = re.sub(r'(\W)\1+', r'\1', text)

    # repeated words
    text = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text)

    # e.g. repeated whitespace
    for repl in replacements_re:
        text = repl[0].sub(repl[1], text)
    return text

# base assumptions are: there is a 'text' column/field
# name of id column may vary
# We load the datafiles from GCS one by one
def load_datafiles_from_gcs(gcs_dir, out_dir, separator=None):
    client = storage.Client.from_service_account_json('../gsa.json')
    bucket = client.get_bucket(gcs_dir)
    blobs = bucket.list_blobs()
    existing_files = set(blob.name for blob in bucket.list_blobs(prefix=out_dir))
    blobs = [blob for blob in blobs if blob.name not in existing_files]

    for blob in blobs:
        _, file_extension = os.path.splitext(blob.name)
        if file_extension in ['.json', '.jsonl']:
            blob.download_as_file(blob.name)
            with open(blob.name, 'r') as json_file:
                json_list = list(json_file)
            res = []
            for json_str in json_list:
                res.append(json.loads(json_str))
            os.remove(blob.name)
            yield 'json', blob.name, res
        elif file_extension == '.parquet':
            res = pd.read_parquet(blob.download_as_bytes())
            yield 'parquet', blob.name, res
        elif file_extension == '.txt':
            # read all lines of the file with an optional seperator
            # if seperator regex not provided, default to '\n'
            blob.download_as_file(blob.name)
            with open(blob.name, 'r') as txt_file:
                bulk = txt_file.read()
                if separator is not None:
                    lines = bulk.split(separator)
                else:
                    lines = bulk.splitlines()
            os.remove(blob.name)
            yield 'txt', blob.name, lines


def normalise_data(data, file_type, incoming_file_name):
    id_field = meta_data[incoming_file_name]['id_field']
    text_field = meta_data[incoming_file_name]['text_field']

    normalised_data = []
    word_count = 0
    if file_type in ['parquet', 'json']:
        for k,entry in enumerate(data):
            if isinstance(entry[text_field], str) and len(entry[text_field]) > 0:
                cleaned_text = clean_text(entry[text_field])
                cid = str(entry[id_field]) if id_field is not None else f"{incoming_file_name}_{str(k)}"
                word_count += len(cleaned_text.split())
                normalised_entry = {
                    "id": cid,
                    "text": cleaned_text,
                    "source": entry.get("source", "not available"),
                    "approx_token_counts_original": entry.get("approx_token_counts_original", -1),
                    "approx_token_counts_translated": entry.get("approx_token_counts_translated", -1)
                }
                normalised_data.append(normalised_entry)
    else:
        for k, txt in enumerate(data):
            cleaned_text = clean_text(txt)
            word_count += len(cleaned_text.split())
            normalised_entry = {
                "id": f"{incoming_file_name}_{str(k)}",
                "text": cleaned_text,
            }
            normalised_data.append(normalised_entry)

    return normalised_data, word_count

def data_transfer(data, output_dir):
    # write to local
    local_file_path = '../tmp/normalised_data.jsonl'
    with open(local_file_path, 'w') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')

    # upload to GCS
    client = storage.Client()
    bucket = client.get_bucket(output_dir)
    blob = bucket.blob(os.path.basename(local_file_path))
    blob.upload_from_filename(local_file_path)

    # remove local file
    os.remove(local_file_path)
    return True

# We then normalise the data and save it to a new .json and upload to the GCS normalised data folder
# We then remove the original data file from the local drive

if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--gcs_dir", type=str, required=True)
    argparser.add_argument("--output_dir", type=str, required=True)
    args = argparser.parse_args()

    file_iterator = load_datafiles_from_gcs(args.gcs_dir.strip("gs://"), args.output_dir.strip("gs://"))

    for file_type, file_name, data in file_iterator:
        normalised_data, word_count = normalise_data(data, file_type, file_name)
        data_transfer(normalised_data, args.output_dir)
        print(f"Normalised {file_name}, Wordcount: {word_count}.")
