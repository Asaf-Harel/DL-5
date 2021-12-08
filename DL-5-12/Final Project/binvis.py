import requests

req = requests.get('https://binvis.io/#/')

with open('index.html', 'w') as file:
    file.write(req.text)

