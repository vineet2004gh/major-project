import requests

def get_user_state():
    # For now, use dummy or IP-based
    try:
        response = requests.get("https://ipapi.co/json/").json()
        return response.get("region", "Maharashtra")  # fallback
    except:
        return "Maharashtra"
