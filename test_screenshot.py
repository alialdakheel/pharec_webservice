from screenshot import collect_image, get_domain
url = 'https://google.com'
url2 = 'https://bbc.co.uk'

collect_image(url)
print(get_domain(url), get_domain(url2))
