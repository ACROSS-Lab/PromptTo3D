import requests
# URL
url = "http://11.0.31.197:8000/generate3dmodel" # Use the Across IP adress locally 

# The prompt

data = {
    "prompt": " a running horse"
}

# send the request POST

response = requests.post(url, json=data)
    
# verification of the sucess state of the request

if response.status_code == 200:
    response_data = response.json()
    download_url = response_data["download_url"]

    # Download the 3D asset
    model_response = requests.get(download_url)
    if model_response.status_code == 200:
        #Save locally the 3D file
        with open("modelCRM.zip", 'wb') as file:
            file.write(model_response.content)
        print("Model successfuly downloaded.")
    else:
        print("Downloading failed.")
else:
    print("Generation process failed:", response.text)