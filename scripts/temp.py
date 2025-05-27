import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import time
import functools
import argparse

from dotenv import load_dotenv

import config

load_dotenv()
BASE_DIR = os.getenv("argus24_base")

# Default settings
process_par = False
map_misspelling = True
write_out = True
sample = False # for testing purposes

# Set up argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process radiology reports for medical condition detection')

    # Add arguments for the variables you want to set
    parser.add_argument('--target', nargs='+', default=['stenosis', 'plaque', 'ischemie', 'hartfailure'],
                        help='List of target conditions to search for')
    parser.add_argument('--radio_path', type=str, required=True, help='Path to the radiology reports parquet file')
    parser.add_argument('--id_col', type=str, default='studyId_0771',
                        help='Name of the ID column in the dataset')
    parser.add_argument('--date_col', type=str, default='onderzoeks_dt',
                        help='Name of the date column in the dataset')
    parser.add_argument('--text_col', type=str, default='content_attachment1_plain_data',
                        help='Name of the text content column in the dataset')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Output folder for results')
    parser.add_argument('--process_par', action='store_true', default=False,
                        help='Enable parallel processing')
    parser.add_argument('--no_map_misspelling', action='store_false', dest='map_misspelling', default=True,
                        help='Disable misspelling mapping')
    parser.add_argument('--no_write_out', action='store_false', dest='write_out', default=True,
                        help='Disable writing results to files')
    parser.add_argument('--sample', action='store_true', default=False,
                        help='Use a small sample for testing')

    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()

# Set variables from arguments
target = args.target
misspelling_file = '../../../_input/synonyms_syntactic.csv'
RADIO_PATH = args.radio_path
ID_COL = args.id_col
DATE_COL = args.date_col
TEXT_COL = args.text_col
OUTPUT_FOLDER = args.output_folder
process_par = args.process_par
map_misspelling = args.map_misspelling
write_out = args.write_out
sample = args.sample

def load_data(radio_path, id_col, date_col, text_col):
    """Load and prepare the radiology data"""
    print(f"Loading data from {radio_path}...")
    radio_txt = pd.read_parquet(radio_path)
    radio_txt = radio_txt[[id_col, date_col, text_col]]
    radio_txt = radio_txt.rename(columns={text_col: 'txt'})
    radio_txt = radio_txt.dropna(subset=['txt'])
    radio_txt = radio_txt.assign(onderzoeks_dt=radio_txt[date_col].dt.date)
    print(f"Loaded {len(radio_txt)} records")
    return radio_txt


if map_misspelling:
    import re
    pattern = re.compile('[\W_\-]+')
    surr = re.compile(r"([\W]*)[\w]+([\W]*)")

    synonym_map = defaultdict(list)
    miss_df = pd.read_csv(misspelling_file, sep=';')
        # go through seed's and add words to the synoms list
    for _seed in miss_df.seed.unique().tolist():
        synonym_map[_seed] = miss_df.loc[miss_df.seed==_seed, 'word'].apply(lambda x: x.strip()).unique().tolist()
    synonym_map_inverted = {_v: k for k,v in synonym_map.items() for _v in v}
    def synonym_map_inverted_fun(x):
        s = surr.findall(x)
        try:
            lead =s[0][0]; trail=s[0][1]
        except IndexError as e:
            lead = ""; trail = ""
        try:
            return lead+synonym_map_inverted[pattern.sub("", x)]+trail
        except:
            return x

common_list = [r'r[ue]st[\s\-]?perfusie',
              r'perfusie[\-\s]?beelden',
              r'perfusie[\-\s]?scan',
              r'r[ue]st[\s\-]?perfusie',
              r'perfusie[\-\s]?opnamen',
              'ostiale', 'danwel', 'objectiveerbare', 'verhaal van',
               'aandoende', 'ruim', 'natief', 'diffuus', 'tijdens',
               'stress', 'als', 'rust', 'waren', 'hemodynamisch',
               'onzekere', 'vervaardigd', 'waarop', 'intermediair']

neg_noise_words = list(set(config.neg_noise_words+common_list))
pos_noise_words = list(set(config.pos_noise_words+common_list))

neg_noise_words = [w.strip() for w in neg_noise_words]
pos_noise_words = [w.strip() for w in pos_noise_words]

# you want to remove ALL terms from the noise lists that are also in the marker lists
collect_pos_marker_words = [_w.lower() for _words in config.pos_markers
                             for _w in _split(_words,
                                              ["?","[","]",",",";",
                                               ".","+","*","^","<"],
                                              [" ", r"\s"])]
collect_neg_marker_words = [_w.lower() for _words in config.neg_markers
                             for _w in _split(_words,
                                              ["?","[","]",",",";",
                                               ".","+","*","^","<"],
                                              [" ", r"\s"])]

pos_noise_words = [_w for _w in pos_noise_words if any(_w in _p for _p in collect_pos_marker_words)==False]
neg_noise_words = [_w for _w in neg_noise_words if any(_w in _p for _p in collect_neg_marker_words)==False]

prior_delimiters = [',', r'\;', r'\:', r'\.', r'\|']

reps = [(re.compile(r'[\r\n]+'), ' '),
        (re.compile(r'\.{2,}'), ' '),
        (re.compile(r'\-{2,}'), ' '),
        (re.compile(r'\s\w\s'), ' '),
        (re.compile(r'[\.\:\;]'), ' . '),
        (re.compile(r'[\+\_\^\$\#\@\;\(\)\{\}\[\],]'), ' '),
        (re.compile(r'\t+'), ' '),
        (re.compile(r'[\W]ong\.'), ':'),
        (re.compile(r'[\W]syst\.'),''),
        (re.compile(r'[23][dD]'), ''),
        (re.compile(r'\([45][Cc][Hh]\)'), ''),
        (re.compile(r'meer dan'), '>'),
        (re.compile(r'minder dan'), '<'),
        (re.compile(r'ongeveer'), ':'),
        (re.compile(r'minimaal'), '>'),
        (re.compile(r'maximaal'), '<'),
        (re.compile(r'minimaal'), '>'),
        (re.compile(r'maximaal'), '<'),
        (re.compile(r'ongeveer'), ':'),
        (re.compile(r'circa'), ':'),
        (re.compile(r'[\=]'), ':'),
        (re.compile(r'minimaal'), '>'),
        (re.compile(r'maximaal'), '<'),
        (re.compile(r'\s{2,}'), ' ')]

def parse_data(df_ct, sample=False):
    def get_agg(x, s='biggest'):
        if s=='biggest':
            if (len(x[0])>len(x[1])) & (len(x[0])>len(x[2])):
                return x[0]
            elif (len(x[1])>len(x[0])) & (len(x[1])>len(x[2])):
                return x[1]
            else:
                return x[2]
        elif s=='concat':
            return "|".join(x)


    df_ct['scan_type'] = 'ct'

    # filter out reports not related to CORONAIREN/STRESS
    old_shape_ct = df_ct.shape[0]

    df_ct.fillna("", inplace=True)
    df_ct = df_ct[~df_ct.txt.str.contains(r'(restenose)|(restenosis)|(graft)|(stent)|(dotter)', case=False)]

    print('We drop {} from the CT radio reports'.format(old_shape_ct-df_ct.shape[0]))

    df  = df_ct[['studyId_0771', 'onderzoeks_dt', 'txt']]

    #df.dropna(subset=['new'], axis=0, inplace=True)
    #df.new = df.new.astype(str)
    #df.new = df.new.str.replace(r"[\+\_\^\$\#\@\±]", "")
    #df.drop(['reporttxt', 'tekst', 'plattetext', 'ONDERZDAT'], axis=1, inplace=True)

    #df.to_csv('text_data/'+dataset+'_radio_filtered_'+scan_type+'.csv', sep=";", index=False)

    print('RADIO -- CT len:{} with {} patients'.format(df_ct.shape[0],
                                                                        df_ct.studyId_0771.nunique()))


    if map_misspelling:
        df = df.assign(txt=df.txt.apply(lambda x: " ".join(map(synonym_map_inverted_fun, x.split()))))

    df['txt_len'] = df.txt.apply(lambda x: len(x))

    df  = df.assign(txt=df.txt.apply(lambda x: config.lspace.sub(" ", x)))
    df  = df.assign(txt=df.txt.apply(lambda x: config.tspace.sub(" ", x)))
    df  = df.assign(txt=df.txt.apply(lambda x: config.mspace.sub(" ", x)))
    df  = df.assign(txt=df.txt.apply(lambda x: config.mdash.sub("-", x)))
    df  = df.assign(txt=df.txt.apply(lambda x: config.mdot.sub(".", x)))

    if sample:
        return df.sample(100)
    else:
        return df

def timer(func):
    # https://realpython.com/primer-on-python-decorators/
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


class clean:
    def __init__(self, replacements=[(re.compile(r'[^\w]{2,}'), ' ')], stopwords=[], lower=True, html=False):
        self.replacements = replacements
        self.stopwords = stopwords
        self.lower = lower
        self.html = html
        assert isinstance(stopwords, list), "stopwords is not a list.."

    def remove_stopwords(self, wlist):
        res = []
        for idx, _w in enumerate(wlist):
            if _w not in self.stopwords:
                res.append(_w)
        return " ".join(res)

    def clean_text(self, txt):
        '''
            remove : list of tuples with compiled regex and replacement
        '''
        print("Processing replacements..., txt type {}".format(type(txt)))
        try:
            for _regtup in self.replacements:
                txt = txt.str.replace(_regtup[0], _regtup[1], regex=True)
        except AttributeError as e:
            msg = "Error with tuple {},\n\t txt.type {}\n\t e: {}".format(_regtup, type(txt), e)
            raise Exception('AttributeError', msg)

        if self.lower:
            print("Processing lowercasing...")
            txt = txt.str.lower()
        if self.html:
            print("Processing html decoding...")
            soup_fun = lambda x: soup(x, 'lxml').text
            txt = txt.apply(soup_fun)

        if (isinstance(self.stopwords, list)) & (len(self.stopwords)>0):
            print("Processing stopwords list")
            stopword_replace = lambda x: self.remove_stopwords(re.split(' ', x))
            txt = txt.apply(stopword_replace)
        return txt

    @timer
    def par_clean_text(self, df, num_processes=None, col_to_process='text'):
        ''' Apply a function separately to each column in a dataframe, in parallel.'''
        if num_processes==None:
            num_processes = min(df.shape[1], cpu_count())
        with Pool(num_processes) as pool:
            seq = df[col_to_process]
            chunk = int(len(seq)/num_processes)
            stuple = tuple([seq[i*chunk:(i+1)*chunk] for i in range(0, num_processes)])
            print("Start multi-processing with {} processes...".format(num_processes))
            results_list = pool.map(self.clean_text, stuple)
            return pd.concat(results_list, sort=True, axis=0)

# sequentially
@timer
def run_me(func):
    return func

def replace_terms(df, process_par=False):
    # remove noise terms (case sensitive)
    cl = clean(replacements=reps, lower=False, html=False, stopwords=[])
    df_new = df.copy()
    if process_par == False:
        # sequentially
        df_new['txt']= run_me(cl.clean_text(df.txt))
    else:
        # in parallel
        df_new['txt'] = cl.par_clean_text(df, num_processes=4, col_to_process = 'txt')

    df_new.dropna(subset=['txt'], axis=0, inplace=True)
    return df_new

# https://stackoverflow.com/questions/4697006/python-split-string-by-list-of-separators
def _split(txt, rems, seps):
    for rem in rems:
        txt = txt.replace(rem,"")
    default_sep = seps[0]
    for sep in seps[1:]:
        txt = txt.replace(sep, default_sep)
    return [i.strip() for i in txt.split(default_sep)]

def get_regex(target):
    if target == 'perfusion':
        synoms = config.synom_dict['perfusion'] # expand

        # ignore other diagnoses in sequences:
        other_ignore = [_v for k,v in config.synom_dict.items() for _v in v if k!='perfusion']
    elif target == 'delayed_enhancement':
        synoms = config.synom_dict['delayed_enhancement'] # expand

        # ignore other diagnoses in sequences:
        other_ignore =  [_v for k,v in config.synom_dict.items() for _v in v if k!='delayed_enhancement']
    elif target == 'stenosis':
        synoms = config.synom_dict['stenosis'] # expand

        # ignore other diagnoses in sequences:
        other_ignore =   [_v for k,v in config.synom_dict.items() for _v in v if k!='stenosis']
        other_ignore += other_ignore + ['stenosegraad', 'coronaire']

    elif target == 'plaque':
        # verkalkte plaque is principe een beter synoniem voor atherosclerose
        synoms = config.synom_dict['plaque'] # expand

        # ignore other diagnoses in sequences:
        other_ignore =  [_v for k,v in config.synom_dict.items() for _v in v if k!='plaque']

    elif target == 'ischemie': # vaatlijden
        synoms = config.synom_dict['ischemie']# expand

        # ignore other diagnoses in sequences:
        other_ignore =  [_v for k,v in config.synom_dict.items() for _v in v if k!='ischemie']

    elif target == 'hartfailure':
        # mine for any and all types of heart failure
        synoms = config.synom_dict['hartfailure']

        other_ignore =  [_v for k,v in config.synom_dict.items() for _v in v if k!='hartfailure']

    elif target == 'thrombi':
        # mine for any and all types of heart failure
        synoms = config.synom_dict['thrombi']

        other_ignore =  [_v for k,v in config.synom_dict.items() for _v in v if k!='thrombi']
    elif target == 'congenitaal':
        synoms = config.synom_dict['congenitaal']

        other_ignore =  [_v for k,v in config.synom_dict.items() for _v in v if k!='congenitaal']
    elif target == 'agatston':
        synoms = [r'agatston\s?score', r'kalk\s?score', r'calcium\s?score', r'agatston\-score', 'agatston'] # expand
        other_ignore = [r'\svan',
                        r'\smet'
                        r'\sbedraagt',
                        r'\sbetreft',
                        r'\sis',
                        r'\svolgens',
                        r'\:',
                        r'\;',
                        r'\=',
                        'meer dan', ',minder dan', 'minimaal', 'maximaal', 'ongeveer']
        # filter, search for [synoms]\s([0-9]+\.?[0-9]*), extract num vals to list, process list per sample

    return other_ignore, synoms

def remove_noise(df, pos_noise_terms, neg_noise_terms):
    # remove noise terms (case sensitive)
    regString = r''+"|".join([r"([\W]"+term+r"[\W])" for term in neg_noise_terms])
    noiseREX = re.compile(regString, re.IGNORECASE)

    dfneg = df
    dfneg.loc[:, 'new'] = dfneg.txt.str.replace(pat=noiseREX, repl="", regex=True)
    dfneg.loc[:, 'new'] = dfneg.txt.apply(lambda x: config.mspace.sub(" ", x))
    print("Removed neg noise terms")

    # remove noise terms (case sensitive)
    regString = r''+"|".join([r"([\W]"+term+r"[\W])" for term in pos_noise_terms])
    noiseREX = re.compile("|".join([r"([\W]"+term+r"[\W])" for term in pos_noise_terms]), re.IGNORECASE)

    dfpos = df
    dfpos.loc[:, 'new'] = dfpos.txt.str.replace(pat=noiseREX, repl="", regex=True)
    dfpos.loc[:, 'new'] = dfpos.txt.apply(lambda x: config.mspace.sub(" ", x))
    print("Removed pos noise terms")
    return dfneg, dfpos


def filter_nonsynoms(df_pos, synoms, target, id_col='studyId_0771', date_col='onderzoeks_dt'):
    old_size = df_pos.shape[0]
    synom_index = df_pos.txt.str.contains("|".join(["("+term+")" for term in synoms]), case=False)
    df_nosyn = df_pos[~synom_index]
    df_nosyn[target] = -1
    print("Filtered out {} from {} reports that do not contain the target terms".\
                              format(old_size-df_nosyn.shape[0], old_size))
    df_nosyn.set_index([id_col, date_col], inplace=True)
    return df_nosyn[[target]], synom_index

def apply_regex(df_neg, df_pos, synoms, other_ignore, target, id_col='studyId_0771', date_col='onderzoeks_dt'):
    # filter out samples that do not contain the target synonyms
    old_size = df_neg.shape[0]
    # filter out synoms with question marks
    # [linestart] [synom] [words] ?
    dfneg = df_neg.copy()
    qMarkREX = re.compile(r'\W+('+"|".join(synoms)+r')\s*\?', re.IGNORECASE)
    dfneg.loc[:, 'txt'] = dfneg.copy().txt.str.replace(pat=qMarkREX, repl="", regex=True)
    print("Filtered out synonyms with question marks")

    # filter out other_ignore
    otherIgnREX = re.compile("|".join([r'('+_t+')' for _t in other_ignore]), re.IGNORECASE)
    dfneg.loc[:, 'txt'] = dfneg.copy().txt.str.replace(pat=otherIgnREX, repl="", regex=True)
    dfneg.loc[:, 'txt'] = dfneg.copy().txt.apply(lambda x: config.mspace.sub(" ", x))
    print("Filtered out other synonyms for other targets")

    # [marker] [neg_markers] + [synoms]
    regString = r''+'('+"|".join(["("+term+")" for term in prior_delimiters])+r')\s*('+\
                        "|".join(["("+term+")" for term in config.neg_markers])+r')\s*('+\
                        "|".join(["("+term+")" for term in synoms])+')'
    negREX = re.compile(regString)
    #print(negREX)
    dfneg.loc[:, 'Certainly_Negative'] = dfneg.copy().txt.str.contains(pat=negREX)
    print("Average certain negativity of filtered samples is {}%".format(100*dfneg.Certainly_Negative.mean()))


    #################################################################################
    #################################################################################


    # filter out synoms with question marks
    # [linestart] [synom] [words] ?
    dfpos = df_pos.copy()
    qMarkREX = re.compile(r'\W+('+"|".join(synoms)+r')\s*\?', re.IGNORECASE)
    dfpos.loc[:, 'txt'] = dfpos.copy().txt.str.replace(pat=qMarkREX, repl="", regex=True)
    print("Filtered out synonyms with question marks")

    dfpos.loc[:, 'txt'] = dfpos.copy().txt.str.replace(pat=otherIgnREX, repl="", regex=True)
    dfpos.loc[:, 'txt'] = dfpos.copy().txt.apply(lambda x: config.mspace.sub(" ", x))
    print("Filtered out other synonyms for other targets")

    # [marker] [pos_markers] + [synoms]
    regString = r''+'('+"|".join([r"("+term+")" for term in prior_delimiters])+r')\s*('+\
                                "|".join(["("+term+")" for term in config.pos_markers])+r')\s*('+\
                                "|".join(["("+term+")" for term in synoms])+')'
    posREX = re.compile(regString)
    print(posREX)
    dfpos.loc[:, 'Certainly_Positive'] = dfpos.copy().txt.str.contains(pat=posREX)
    print("Average certain positivity of filtered samples is {}%".format(100*dfpos.Certainly_Positive.mean()))

    #################################################################################
    #################################################################################


    dfneg['negative'] = dfneg['Certainly_Negative'].astype(int)
    dfpos['positive'] = dfpos['Certainly_Positive'].astype(int)

    pos_df = pd.DataFrame(dfpos.groupby([id_col, date_col])\
                            .positive.max())
    neg_df = pd.DataFrame(dfneg.groupby([id_col, date_col])\
                            .negative.max())

    out = pos_df.join(neg_df)
    out['mult'] = (out['positive']*out['negative'])
    res = out # [((out.positive>0) | (out.negative>0))] # arguably :
    res[target] = res[['positive', 'negative']].apply(lambda x: 0 if x['negative']==1
                                                                else 1 if x['positive']==1
                                                                else 2, axis=1)
    return res[[target]]


if __name__ == "__main__":
    # Create the output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Print the configuration being used
    print("\nConfiguration:")
    print(f"Targets: {target}")
    print(f"Radio path: {RADIO_PATH}")
    print(f"ID column: {ID_COL}")
    print(f"Date column: {DATE_COL}")
    print(f"Text column: {TEXT_COL}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Parallel processing: {process_par}")
    print(f"Map misspelling: {map_misspelling}")
    print(f"Write results: {write_out}")
    print(f"Sample mode: {sample}\n")

    # Load the data
    radio_txt = load_data(RADIO_PATH, ID_COL, DATE_COL, TEXT_COL)

    print("\t","+"*30, "Parsing data")
    parsed = parse_data(radio_txt, sample=sample)

    print("\t\t","+"*30, "Replace terms")
    parsed = replace_terms(parsed, process_par=process_par)

    print("\t\t\t","+"*30, "Cleaning data")
    cleaned_data_neg, cleaned_data_pos = remove_noise(parsed, pos_noise_words, neg_noise_words)
    data_dict = {'neg': cleaned_data_neg,  'pos': cleaned_data_pos}

    print("\t\t","+"*30, "Getting specific noise/synoms")
    for _target in target:
        print(f"Processing target: {_target}")
        other_ignore, syns = get_regex(_target)

        print("\t\t\t\t","+"*30, "Applying regex")
        df_neg = data_dict['neg']
        df_pos = data_dict['pos']

        df_nonsyms, synom_index = filter_nonsynoms(df_neg, syns, _target, id_col=ID_COL, date_col=DATE_COL)

        res = apply_regex(df_neg[synom_index], df_pos[synom_index], syns, other_ignore, _target, id_col=ID_COL, date_col=DATE_COL)

        res = pd.concat([res, df_nonsyms], axis=0)
        if write_out:
            # Create target-specific output directory
            target_output_dir = os.path.join(OUTPUT_FOLDER, f"regexLegacy_{_target}")
            os.makedirs(target_output_dir, exist_ok=True)
            output_file = os.path.join(target_output_dir, 'results.csv')
            res.to_csv(output_file, sep=";", index=True)
            print(f"Results for {_target} saved to {output_file}")
