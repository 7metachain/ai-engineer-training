@app.route('/')
def index():
    a = requests.get('http://localhost:5000/')
    b = requests.get('http://localhost:6000/')
    c = requests.get('http://localhost:7000/')
    return a.text + b.text + c.text


from concurrent.futures import ThreadPoolExecutor
@app.route('/')
def index():
    url_list = ['http://localhost:5000/', 'http://localhost:6000/', 'http://localhost:7000/']

    def fetch_url(url):
        return requests.get(url).text

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(fetch_url, url_list)

    return '\n'.join(results)   

