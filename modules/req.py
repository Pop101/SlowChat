import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def post_with_retry(url, **kwargs):
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])

    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    response = session.post(url, **kwargs)
    session.close()
    
    return response