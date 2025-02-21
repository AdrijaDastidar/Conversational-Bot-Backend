# 🤖 Friendly Chatbot Backend  

## Overview  
The **Friendly Chatbot Backend** powers the conversational AI interface by processing user queries and generating intelligent responses. It is built using **Flask** and integrates **LangChain** with **ChatGroq** for advanced conversational capabilities. The backend is hosted at:  
**🌐 Live Backend URL:** [https://conversational-bot-backend.onrender.com](https://conversational-bot-backend.onrender.com/res)  

## ✨ Features  
- 💬 **Natural Language Understanding:** Generates context-aware and coherent responses using the ChatGroq API.  
- 🔄 **Conversational Memory:** Maintains a buffer of recent interactions for better context retention.  
- 🌍 **Cross-Origin Support:** CORS-enabled for seamless integration with frontend applications.  
- ⚡ **Fast and Reliable:** Hosted with Waitress for production-grade performance.  

## 🛠️ Technology Stack  
- **Framework:** Flask  
- **Language Model:** LangChain with ChatGroq API  
- **Memory Management:** ConversationBufferWindowMemory  
- **Hosting:** Waitress  

## 🚀 Setup and Installation  

### 1️⃣ Clone the Repository  
```bash  
git clone https://github.com/your-username/friendly-chatbot-backend.git  
cd friendly-chatbot-backend  
```  

### 2️⃣ Set Up Environment Variables  
Create a `.env` file in the root directory and add the following:  
```env  
GROQ_API_KEY=your-groq-api-key  
```  

### 3️⃣ Install Dependencies  
Ensure you have Python installed, then set up a virtual environment and install the required packages:  
```bash  
python -m venv venv  
source venv/bin/activate  # On Windows, use venv\Scripts\activate  
pip install -r requirements.txt  
```  

### 4️⃣ Run the Backend  
To start the backend server locally:  
```bash  
python app.py  
```  
The backend will be available at `http://localhost:8080`.  

### 5️⃣ Live Deployment  
The live backend is hosted at:  
**[https://conversational-bot-backend.onrender.com](https://conversational-bot-backend.onrender.com/res)**  

## 📂 Project Structure  
```plaintext  
.  
├── app.py                  # Main Flask application  
├── prompt.py               # System prompt configuration  
├── requirements.txt        # Dependencies  
├── src/                    # Static and template files (if needed)  
├── .env                    # Environment variables  
└── README.md               # Documentation  
```  

## 💡 Usage  

### 🔗 API Endpoint  
#### POST `/res`  
Handles user queries and returns bot responses.  

- **Request:**  
  ```json  
  {  
    "query": "Hello, how are you?"  
  }  
  ```  
- **Response:**  
  ```json  
  {  
    "response": "I'm just a bot, but I'm doing great! How can I assist you today?"  
  }  
  ```  

### 🔄 CORS Configuration  
The backend is configured to allow requests from `http://localhost:5173` (frontend). Update the `Access-Control-Allow-Origin` header in `app.py` to whitelist other domains if needed.  

## 🔧 Customization  

### 🔑 API Key  
Ensure you have a valid **GROQ_API_KEY** in your `.env` file.  

### 🧠 Conversational Memory  
The memory buffer length can be adjusted by modifying `conversational_memory_length` in `app.py`.  

### 🔥 Temperature and Max Tokens  
Customize the response generation parameters in the `ChatGroq` initialization:  
- `temperature`: Controls randomness of responses.  
- `max_tokens`: Sets the maximum length of generated responses.  

## 🌟 Future Enhancements  
- 📊 **Analytics Integration**: Add support for chat logging and analytics.  
- 🌐 **Multi-Language Support**: Expand capabilities for multi-lingual conversations.  
- 🔌 **Voice Interaction**: Enable speech-to-text and text-to-speech features.  
- 🔐 **Authentication**: Add user authentication for secure access.  

## 🤝 Contributing  
1. Fork the repository.  
2. Create a feature branch: `git checkout -b feature-name`.  
3. Commit your changes: `git commit -m 'Add feature'`.  
4. Push the branch: `git push origin feature-name`.  
5. Open a Pull Request.   

## 🙏 Acknowledgments  
- Thanks to **Flask**, **LangChain**, and **ChatGroq** for making this project possible.  
- Inspired by the goal of creating smarter, user-friendly chatbots.  

Enjoy building and scaling the Friendly Chatbot Backend! 🚀
