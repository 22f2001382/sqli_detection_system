import secrets,hashlib,hmac
def hash_api_key(api_key):
    return hashlib.sha256(api_key.encode()).hexdigest()

def generate_api_key():
    return secrets.token_hex(32)  # 64-character secure API key

def validate_api(api_key,api_key_hash):
    return hmac.compare_digest(hash_api_key(api_key), api_key_hash)
