import requests
from tqdm import tqdm


def download_file_from_url(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(filename, "wb") as handle:
        for data in tqdm(response.iter_content(), total=total_size):
            handle.write(data)


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32*1024
    total_size = int(response.headers.get('content-length', 0))

    with tqdm(desc=destination, total=total_size, unit='B',
              unit_scale=True) as pbar:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    pbar.update(CHUNK_SIZE)
                    f.write(chunk)
    #
    # CHUNK_SIZE = 32768
    #
    # with open(destination, "wb") as f:
    #     for chunk in response.iter_content(CHUNK_SIZE):
    #         if chunk: # filter out keep-alive new chunks
    #             f.write(chunk)
