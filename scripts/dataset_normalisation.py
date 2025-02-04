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
from typing import List, Tuple

replacements = [
    (r'\s{2,}', ' ')
]
replacements_re = [(re.compile(repl[0]), repl[1]) for repl in replacements]

meta_data = {
    'MariaNMT_cardiomyopathy.json': {
                                        'id_field': 'id',
                                        'text_field': 'text'
                                    },
    'MariaNMT_acute_coronary_syndrome.jsonl': {
                                        'id_field': 'id',
                                        'text_field': 'text'
                                    },
    'MariaNMT_atrial_fibrillation.jsonl': {
                                        'id_field': 'id',
                                        'text_field': 'text'
                                    },
    'MariaNMT_cardiomyopathy.jsonl': {
                                        'id_field': 'id',
                                        'text_field': 'text'
                                    },
    'MariaNMT_cardiovascular_disease.jsonl': {
                                        'id_field': 'id',
                                        'text_field': 'text'
                                    },
    'PubmedPatients_nllb200.json':{
                                     'id_field':'patient_uid',
                                      'text_field':'text'
                                    },
    'meditron_guidelines_gpt4omini.json': {
                                    'id_field':'id',
                                    'text_field':'text'
    },
    'DeepL_PUBMED_ABSTRACTS_*.parquet': {
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
    'PMC-Patients-V2_GeminiFlash-1-5_corrected.json': {
                                    'id_field': 'patient_uid',
                                    'text_field': 'text'
    },
    'Scraped.jsonl': {
                                    'id_field': None,
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
    },
    'GeminiFlash15_mimic3_*.jsonl': {
                                    'id_field': 'id',
                                    'text_field': 'text'
    },
    'AGCT_MariaNMT.jsonl': {
                                    'id_field': 'id',
                                    'text_field': 'text'
    },
    'Apollo_medicalGuideline_en_text_GeminiFlash15.json':{
                                    'id_field': 'id',
                                    'text_field': 'text'
    },
    'EMEA_*.jsonl' :{
                                    'id_field': None,
                                    'text_field': 'text'
    },
    'MariaNMT_Mimic4_*.json' : {
                                    'id_field': 'id',
                                    'text_field': 'text'
    },
    'MariaNMT_mimic3_*.json' : {
                                    'id_field': 'id',
                                    'text_field': 'text'
    },
     'meditron_guidelines_gpt4omini.json': {
                                    'id_field': 'id',
                                    'text_field': 'text'
    },
    'ntvg.parquet': {
                                   'id_field': None,
                                   'text_field': 'total_text'
    },
    'apollo_books_nllb200.jsonl': {
                                'id_field': 'id',
                                'text_field': 'text'
    },
    'wikipedia-*.parquet': {
                            'id_field': None,
                            'text_field': 'text'
    },
    'apollo_wiki_mariaNMT.jsonl': {
                            'id_field': 'id',
                            'text_field': 'text'
    },
    'GPT4omini_pubmed_*.jsonl': {
                            'id_field': 'id',
                            'text_field': 'text'
    }
}

def regexifyer(tdict: dict)->List[Tuple[str,str]]:
    '''
        Given a dictionary with keys that contain catchall * and ? characters
        we will return a list of tuples with the regexified key and the value.
        This also means: escaping existing regex-special characters
    '''
    return [(re.sub(r'\*', '.*', key.replace('.', '[.]')), key) for key in tdict.keys()]

meta_keys_re = regexifyer(meta_data)

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
    bucket = client.get_bucket(gcs_dir.split("/")[0])
    blobs = bucket.list_blobs(prefix=gcs_dir.split("/")[1])
    existing_files = set(blob.name for blob in bucket.list_blobs(prefix=out_dir.split("/")[1]))

    print(f"Currently existing files {existing_files}")

    namer = lambda x: x.name.split('/')[-1].split('.')[0]

    blobs = [blob for blob in blobs if  not any([namer(blob) in _existing_file for _existing_file in existing_files])]

    failed_lines = []
    fix_needed = []
    print(f"Downloading files from GCS")
    for blob in blobs:
        _, file_extension = os.path.splitext(blob.name)
        filename = blob.name.split('/')[-1]
        tmp_file_name = '../tmp/'+filename
        print(f"Downloading {filename}")
        if file_extension in ['.json', '.jsonl']:
            with open(tmp_file_name, 'wb') as file_out:
                blob.download_to_file(file_out)
            with open(tmp_file_name, 'r') as json_file:
                json_list = list(json_file)
            res = []
            for ln, json_str in tqdm.tqdm(enumerate(json_list)):
                try:
                    res.append(json.loads(json_str))
                except:
                    # fix attempt, simply add \"}
                    json_str += '\"}'
                    fix_needed.append(ln)
                    try:
                        res.append(json.loads(json_str))
                    except:
                        failed_lines.append(ln)
            print(f"{len(fix_needed)} lines had json-parsing issue")
            print(f"Failed to load {len(failed_lines)} lines, after simple fix attempt")
            os.remove(tmp_file_name)
            yield 'json', filename, res
        elif file_extension == '.parquet':
            with open(tmp_file_name, 'wb') as file_out:
                blob.download_to_file(file_out)
            res = pd.read_parquet(tmp_file_name)
            os.remove(tmp_file_name)
            yield 'parquet', filename, res
        elif file_extension == '.txt':
            # read all lines of the file with an optional seperator
            # if seperator regex not provided, default to '\n'
            with open(tmp_file_name, 'wb') as file_out:
                blob.download_to_file(file_out)
            with open(tmp_file_name, 'r', encoding='latin1') as txt_file:
                bulk = txt_file.read()
                if separator is not None:
                    lines = bulk.split(separator)
                else:
                    lines = bulk.splitlines()
            os.remove(tmp_file_name)
            yield 'txt', filename, lines


def normalise_data(data, file_type, incoming_file_name):

    # find the right meta_id using the meta_keys_re
    #
    id_field = None
    text_field = None
    for key_re, _key in meta_keys_re:
        if re.match(key_re, incoming_file_name) is not None:
            id_field = meta_data[_key]['id_field']
            text_field = meta_data[_key]['text_field']
            print(f"Using meta key:{_key}")
            break
        #print(f'NO MATCH \t {key_re}, {_key}\n')

    if not text_field and file_type!='txt':
        raise ValueError(f"Could not find the right meta data for {incoming_file_name}")

    normalised_data = []
    word_count = 0
    if file_type in ['json']:
        for k, entry in tqdm.tqdm(enumerate(data)):
            if entry is not None:
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
    elif file_type in ['parquet']:
        for index, row in tqdm.tqdm(data.iterrows()):
            if isinstance(row[text_field], str) and len(row[text_field]) > 0:
                cleaned_text = clean_text(row[text_field])
                cid = str(row[id_field]) if id_field is not None else f"{incoming_file_name}_{str(index)}"
                word_count += len(cleaned_text.split())
                normalised_entry = {
                    "id": cid,
                    "text": cleaned_text,
                    "source": row.get("source", "not available"),
                    "approx_token_counts_original": row.get("approx_token_counts_original", -1),
                    "approx_token_counts_translated": row.get("approx_token_counts_translated", -1)
                }
                normalised_data.append(normalised_entry)
    else:
        for k, txt in tqdm.tqdm(enumerate(data)):
            if txt is not None and len(txt) > 0:
                cleaned_text = clean_text(txt)
                word_count += len(cleaned_text.split())
                normalised_entry = {
                    "id": f"{incoming_file_name}_{str(k)}",
                    "text": cleaned_text,
                }
                normalised_data.append(normalised_entry)

    return normalised_data, word_count

def data_transfer(data, output_dir, file_name):
    # write to local
    local_file_path = '../tmp/normalised_data.jsonl'
    with open(local_file_path, 'w') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')

    # upload to GCS
    client = storage.Client.from_service_account_json('../gsa.json')
    bucket = client.get_bucket(output_dir.split("/")[0])
    blob = bucket.blob(output_dir.split("/")[1] + '/' + file_name + '_normalised.json')
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
        data_transfer(normalised_data, args.output_dir.strip("gs://"), file_name)
        print(f"Normalised and uploaded {file_name}, Wordcount: {word_count}.")
