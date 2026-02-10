import httpx
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY

headers = {
    "apikey": SUPABASE_SERVICE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    "Content-Type": "application/json",
}

resp = httpx.post(
    f"{SUPABASE_URL}/auth/v1/admin/users",
    headers=headers,
    json={
        "email": "moussa.saidi.01@gmail.com",
        "password": "testpassword123",
        "email_confirm": True,
    },
)
print(f"Status: {resp.status_code}")
data = resp.json()
if resp.status_code == 200:
    print(f"User created: id={data['id']}  email={data['email']}")
else:
    print(f"Error: {data}")
