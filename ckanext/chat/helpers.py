import ckan.plugins.toolkit as toolkit
import requests


def service_available():
    url = toolkit.config.get("ckanext.chat.completion_url")
    if not url:
        return False
    try:
        # Perform a HEAD request (lightweight check) to see if the service responds
        response = requests.head(url, timeout=5, verify=False)
        if (200 <= response.status_code < 400) or response.status_code == 405:
            return True  # URL is reachable and returns a valid status code
        else:
            return False  # URL is reachable but response status is not valid
    except requests.RequestException as e:
        # If there's any issue (timeout, connection error, etc.)
        return False


def get_helpers():
    return {
        "service_available": service_available,
    }
