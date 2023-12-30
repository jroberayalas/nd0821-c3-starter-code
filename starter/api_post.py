import requests
import json

def make_api_request(url, data):
    """
    Send a POST request to the specified URL with the given data.

    :param url: The URL of the API endpointendpoint.
    :param data: A dictionary containing the data to be sent in the POST request.
    :return: A tuple of (status code, response JSON).
    """
    response = requests.post(url, json=data)
    return response.status_code, response.json()

# Replace this with the URL of your Render app and appropriate endpoint
api_url = "https://application-platform.onrender.com/predict"

# Replace this with the data you want to send to your API
# The structure of this data depends on your specific model's requirements
data = {
        "age": 58,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 93664,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    }

status_code, response = make_api_request(api_url, data)

print(f"Status Code: {status_code}")
print(f"Response: {json.dumps(response, indent=2)}")
