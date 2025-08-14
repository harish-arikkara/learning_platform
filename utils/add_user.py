from handle_user import create_user, get_user

# Sample data
USER_ID = "testuser"
NAME = "Test User"
PASSWORD = "password123"
EMAIL = "test@example.com"
FIRM = "Sample Firm"
UNIT = "R&D"
LOCATION = "Remote"

# Create the user
create_user(
    user_id=USER_ID,
    name=NAME,
    password=PASSWORD,
    email=EMAIL,
    firm=FIRM,
    unit=UNIT,
    location=LOCATION
)

# Confirm creation
user = get_user(USER_ID)
print("Inserted user:", user)
print("You can now log in with:")
print(f"User ID: {USER_ID}")
print(f"Password: {PASSWORD}")
