from flask import Flask, jsonify, request

app = Flask(__name__)

# Sample data: a list of users
users = [
    {"id": 1, "name": "John Doe", "email": "john@example.com"},
    {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
]




@app.route('/users', methods=['GET'])
def get_all_users():
    return jsonify(users), 200

@app.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = next((user for user in users if user["id"] == user_id), None)
    if user:
        return jsonify(user), 200
    else:
        return jsonify({"error": "User not found"}), 404

# Route to create a new user
@app.route('/users', methods=['POST'])
def create_user():
    
    name = request.json.get('name')
    email = request.json.get('email')
    # new_user['id'] = len(users) + 1  # Generate new ID
    new_user = {'id':len(users) + 1, 'name':name, 'email':email}
    users.append(new_user)
    return jsonify(new_user), 201

# Route to update a user by ID
@app.route('/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    user = next((user for user in users if user["id"] == user_id), None)
    if user:
        data = request.get_json()
        user.update(data)
        return jsonify(user)
    else:
        return jsonify({"error": "User not found"}), 404

# Route to delete a user by ID
@app.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    global users
    users = [user for user in users if user["id"] != user_id]
    return jsonify({"message": "User deleted"})

if __name__ == '__main__':
    app.run(debug=True)
