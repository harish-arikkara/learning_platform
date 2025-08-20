from handle_user import create_user, get_user

# Sample multiple users
users_data = [
    {
        "user_id": "testuser1",
        "name": "Alice Smith",
        "password": "123secure",
        "email": "alice@example.com",
        "firm": "Sample Firm",
        "unit": "R&D",
        "location": "Remote",
    },
    {
        "user_id": "testuser2",
        "name": "Bob Johnson",
        "password": "secure456",
        "email": "bob@example.com",
        "firm": "Tech Corp",
        "unit": "Marketing",
        "location": "New York",
    },
    {
        "user_id": "testuser3",
        "name": "Charlie Brown",
        "password": "mypassword789",
        "email": "charlie@example.com",
        "firm": "Innovate Ltd",
        "unit": "Design",
        "location": "London",
    }
]

# Create multiple users
for u in users_data:
    create_user(
        user_id=u["user_id"],
        name=u["name"],
        password=u["password"],
        email=u["email"],
        firm=u["firm"],
        unit=u["unit"],
        location=u["location"],
    )

    # Confirm creation
    user = get_user(u["user_id"])
    print("\nInserted user:", user)
    print("You can now log in with:")
    print(f"User ID: {u['user_id']}")
    print(f"Password: {u['password']}")
