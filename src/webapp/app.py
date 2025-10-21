from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import pyrebase
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Firebase configuration
firebaseConfig = {
    "apiKey": "AIzaSyB4H-5X7ISCY-7Zc2mwAXJgAgWxkw7cizM",
    "authDomain": "hand-project2.firebaseapp.com",
    "databaseURL": "https://hand-project2-default-rtdb.firebaseio.com",
    "projectId": "hand-project2",
    "storageBucket": "hand-project2.firebasestorage.app",
    "messagingSenderId": "200301107854",
    "appId": "1:200301107854:web:2642f633f8bed32de787df",
    "measurementId": "G-2F63JNJQWV"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# Firebase Admin SDK initialization
cred = credentials.Certificate("C:\Users\PMLS\Desktop\updated\firebase_1.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': firebaseConfig["databaseURL"]
})

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    if "user" in session:
        return render_template("dashboard.html")
    else:
        flash("Please log in first.", "warning")
        return redirect(url_for("login_page"))

@app.route("/history")
def history():
    return render_template("history.html")

# Route to handle sending a command
@app.route('/send_command', methods=['POST'])
def send_command():
    command = request.form.get('command')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save command and timestamp to Firebase under a unique key in commandHistory
    ref_history = db.reference('commandHistory')
    ref_history.push({
        'command': command,
        'timestamp': timestamp
    })

    # Save the latest command separately in Robo_command
    ref_robo_command = db.reference('Robo_command')
    ref_robo_command.set(command)

    return jsonify(status=f'Command sent and saved: {command}')
# Route to retrieve command history
@app.route('/get_command_history', methods=['GET'])
def get_command_history():
    ref = db.reference('commandHistory')
    history = ref.get()
    history_data = []
    if history:
        for key, command in history.items():
            history_data.append({
                'key': key,
                'command': command['command'],
                'timestamp': command['timestamp']
            })
    return jsonify({'history': history_data})

# Route to delete a command by its key
@app.route('/delete_command/<string:key>', methods=['DELETE'])
def delete_command(key):
    print(f"Attempting to delete command with key: {key}")  # Debugging log
    ref = db.reference(f'commandHistory/{key}')
    ref.delete()
    print(f"Command with key {key} deleted from Firebase.")  # Debugging log
    return jsonify({'status': 'Command deleted successfully'})


# Login and Signup Routes
@app.route("/login_page")
def login_page():
    return render_template("login.html")

@app.route("/signup_page")
def signup_page():
    return render_template("signup.html")

@app.route("/signup", methods=["POST"])
def signup():
    email = request.form["email"]
    password = request.form["password"]

    try:
        auth.create_user_with_email_and_password(email, password)
        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for("login_page"))
    except Exception as e:
        flash("Error creating account. Please try again.", "danger")
        return redirect(url_for("signup_page"))


@app.route("/login", methods=["POST"])
def login():
    email = request.form["email"]
    password = request.form["password"]

    try:
        # Attempt to sign in the user using Firebase authentication
        user = auth.sign_in_with_email_and_password(email, password)

        # Save user info in the session
        session['user'] = user
        flash("Logged in successfully!", "success")

        # After successful login, redirect to the dashboard
        return redirect(url_for("dashboard"))
    except Exception as e:
        flash("Invalid email or password. Please try again.", "danger")
        return redirect(url_for("login_page"))


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully.", "success")
    return redirect(url_for("login_page"))

if __name__ == "__main__":
    app.run(debug=True)