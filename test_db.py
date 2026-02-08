from utils.supabase_client import supabase

result = supabase.table("pricing_tiers").select("*").limit(1).execute()
print("Database connected")
print(result)
