"""
Copyright (C) 2015 Baxter Eaves
License: Do what the fuck you want to public license (WTFPL) V2
Scrape abstracts from the Proceedings of the National Academy of Sciences.
Requires: beutifulsoup4
NOT WRITTEN BY ME ORGINALLY, BUT MODIFIED TO FIT MY PURPOSES (NITHIN)
"""

import re
import sys
import time
import pickle
import requests

from bs4 import BeautifulSoup


HEADERS = {'Accept-Language': 'en-US,en;q=0.8'}
PNAS_URL = "http://www.pnas.org"
PNAS_CONTENT_URL = PNAS_URL + "/content/by/year"

MAX_REQUEST_RATE = 3.1  # PNAS gets upset if you go any faster
TIMEOUT = 60  # PNAS' website is damned slow


def get_abstract(absurl):
    url = absurl

    sys.stdout.write('_%s...' % (url,))
    sys.stdout.flush()

    t_start = time.time()
    page = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    soup = BeautifulSoup(page.text, 'html.parser')

    abspars = soup.find_all("div", "section abstract")
    print(abspars)

    if len(abspars) == 0:
        print("Failed to find abstract.")
    # elif len(abspars) == 1:
    #     abspar = abspars[0]
    # else:
    #     # Assume that the abstract is the paragraph with the most text.
    #     parlens = [len(par.text) for par in abspars]
    #     absidx = max(enumerate(parlens), key=lambda x: x[1])[0]
    #     abspar = abspars[absidx]

    # if len(abspar.text) < len(title):
    #     raise RuntimeError("Abstract short. Maybe found the wrong thing?")
    else:
        abspar = abspars[0]
        with open("abstracts.txt", "a+") as f:
            f.write(abspar.text + "DOC DONE!")
        return abspar.text
    # print(abspar.text)
    # abstract = re.sub('[ \t\n]+', ' ', abspar[0].text)

    # data = {
    #     'Title': title,
    #     'Abstract': abstract,
    #     'Authors': "; ".join(a['content'] for a in authors),
    #     'Date': soup.find("meta", {"name": "DC.Date"})['content'],
    #     'Volume': soup.find("meta", {"name": "citation_volume"})['content'],
    #     'Issue': soup.find("meta", {"name": "citation_issue"})['content'],
    #     'ISSN': "; ".join(n['content'] for n in issn)}

    # t_total = time.time() - t_start
    # t_diff = MAX_REQUEST_RATE-t_total

    # sys.stdout.write('(%1.2f sec)%s\n' % (t_total, title,))
    # sys.stdout.flush()

    # if t_diff > 0:
    #     time.sleep(t_diff)

    # return data


def get_issue_abstract_urls(vol, issue):
    sys.stdout.write('  Issue {}...'.format(issue+1))
    sys.stdout.flush()

    url = PNAS_URL + "/content/{}/{}.toc".format(vol, issue+1)
    print(url)
    page = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    with open("page.txt", "w") as file:
        file.write(page.text)
    soup = BeautifulSoup(page.text, 'html.parser')
    articles = soup.find_all('a')
    abstract_articles = []
    for i in articles:
        if "doi" in i['href']:
            abstract_articles.append(i)
    print(abstract_articles)

    retval = []
    for absurl in abstract_articles:
        #print("st")
        #print(absurl)
        #print("do")
        retval.append([vol, issue, absurl['href']])

    sys.stdout.write('done (%d abstracts).\n' % (len(retval),))
    return retval


def get_abstract_urls_for_year(year):
    sys.stdout.write('Year: {}\n'.format(year))
    sys.stdout.flush()

    url = "{}/{}".format(PNAS_CONTENT_URL, year)
    page = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    # vols = re.findall(r'"/content/vol(\d+)', page.text)
    # vol = vols[0]
    # n_issues = len(vols)

    retval = []
    for issue in range(26):
        retval.extend(get_issue_abstract_urls(98, issue))
    return retval


def genargs(first_year=2001, last_year=2001):
    """ Generate args to scrape all PNAS abstracts from first_year to
    last_year.
    """
    print("Scraping abstract URLS {}-{}".format(first_year, last_year))
    args = []
    for year in range(first_year, last_year+1):
        args.extend(get_abstract_urls_for_year(year))

    argsout = []
    for i, arg in enumerate(args):
        argsout.append(tuple(arg + [i]))

    return argsout


def scrape(args, filename='data/pnas.pkl', start=None):
    """ Scrape PNAS database for abstracts.
    Implementation notes
    --------------------
    - Won't stop unless until it's done or until you ctrl+C.
    - Saves the data to a list of dicts, that can be easily converted into a
    pandas DataFrame.
    Parameters
    ----------
    args : list(tuple)
        Generated by gen_args()
    filename : str
        Filename for the output
    start : int >= 0 or None
        If an int, n, scrape starts at the nth arg and appends new abstracts to
        filename. This is used if there is some unhandled error that occurs,
        e.g. PNAS kicks you out.
    Example
    -------
    Convert to a pandas DataFrame
    >>> import pickle
    >>> import pandas
    >>> args = genargs(first_year=1999, last_year=2001)
    >>> scrape(args, 'pnas.pkl')
    >>> data = pickle.load('pnas.pkl', 'rb')
    >>> df = pandas.DataFrame(data)
    Resume a scrape that failed on the 768th abstract
    >>> '+ Exception: 768 failed'
    >>> scrape(args, 'pnas.pkl', start=768)
    """
    if start is None:
        start = 0
        data = []
    else:
        data = pickle.load(open(filename, 'rb'))

    for _, _, url, idx in args[start:]:
        success = False
        while not success:
            try:
                # save each time we scrape so that we can resume if PNAS kicks
                # us out.
                data.append(get_abstract(url))
                # pickle.dump(data, open(filename, 'wb'))
                success = True
            except TypeError as err:
                print("\n+ TypeError (request failure): %d failed" % (idx,))
                print("+ Waiting for %d seconds before retry..." % (TIMEOUT,))
                time.sleep(TIMEOUT)
            except Exception as err:
                print("\n+ Exception: %d failed" % (idx,))
                raise(err)
    return data


if __name__ == "__main__":
    args_filename = 'data/pnas01_args.pkl'
    data_filename = 'data/pnas01_data.pkl'

    # NOTE: I've found that in general the issue splash pages load
    # significantly slower than the abstract pages. We first scrape the splash
    # pages for # abstract URLs and then scrap those URLs for abstracts and
    # other metadata.
    # NOTE: PNAS is pretty finicky about scraping. They only let you make one
    # request every 3 seconds. One year has about 2400 abstracts, so each year
    # is going to take you two hours, at minimum.
    args = genargs(first_year=2001, last_year=2001)

    pickle.dump(args, open(args_filename, 'wb'))
    args = pickle.load(open(args_filename, 'rb'))

    scrape(args, data_filename)