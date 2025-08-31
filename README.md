# 🗳️ Vote Buddy – Your Personalized Election Assistant

Vote Buddy is an AI-powered election assistance platform designed to make voting simpler and more informed. It helps users discover candidate details, polling station information, and election schedules through a smart chatbot with multilingual support.  

---

## ✨ Features

- 🔹 **AI-Powered Chatbot** – Ask any election-related question and get instant responses.  
- 🔹 **Polling Station Finder** – Locate your nearest polling booth based on your details.  
- 🔹 **Candidate Lookup** – Get details about candidates contesting in your area.  
- 🔹 **Language Support** – Interact in English and Hindi.  
- 🔹 **Clean UI** – Lightweight, fast, and user-friendly design.  
- 🔹 **Secure & Scalable** – Backend powered by Flask and secure APIs.  

---

## 🛠️ Tech Stack

| Layer              | Technology Used                                 |
|--------------------|-------------------------------------------------|
| **Frontend**       | HTML, CSS, JavaScript                           |
| **Backend**        | Flask (Python)                                  |
| **AI Integration** | OpenAI GPT-5 ( Groq API for LLM support), Claude     |
| **Authentication** | Firebase Authentication / JWT                  |
| **Deployment**     | Render                                         |
| **Version Control**| Git & GitHub                                   |
| **APIs**           | Election Commission APIs, Location APIs         |

---

## 🌐 Live Demo

| Platform       | Link                                                   |
|----------------|--------------------------------------------------------|
| **Website**    | [Vote Buddy Live](https://team-project-e4hy.onrender.com) |
| **GitHub Repo**| [team-project](https://github.com/Parth-ctrl490/team-project) |

---

## 📂 Project Structure

team-project/
├── app.py # Main Flask application
├── static/ # JavaScript, CSS, images
├── templates/ # HTML templates
├── chatbot/ # AI chatbot integration code
├── database/ # Database models and logic (optional)
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 🚀 Installation & Setup (Run Locally)

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Parth-ctrl490/team-project.git
   cd team-project


Install Python Dependencies

pip install -r requirements.txt


Set Environment Variables
Create a .env file in your project root:

OPENAI_API_KEY=your_api_key_here
FIREBASE_CONFIG=your_firebase_config_json
SECRET_KEY=your_flask_secret


Run the App

flask run


Visit http://127.0.0.1:5000 to test locally.

🔮 Planned Features

📌 Voter Document Upload & AI Verification

📌 Interactive Maps for Polling Booth Locations

📌 Predictive Assistance for common voter queries

📌 Analytics Dashboard for Admins

👥 Team

Parth-ctrl490 – Lead Developer

📜 License

This project is licensed under the MIT License. See the LICENSE
 file for details.

🙌 Acknowledgments

Built with ❤️ by Parth-ctrl490 & team

Thanks to Render
 for deployment hosting

Powered by OpenAI GPT-5
 & Election APIs

⭐ If you like this project, don’t forget to star the repo!


---
