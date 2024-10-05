# Personal Assistant

This application allows you to teach information to an AI agent and ask questions about the stored knowledge. It runs locally on your machine using LM Studio for the language model server.

## Requirements

* Python 3.8 or newer
* pip (Python package installer)
* Modern web browser
* LM Studio (https://lmstudio.ai/)

## Local Setup Instructions

1. Clone the repository or download and extract the project files to your local machine.

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up LM Studio:
   * Download and install LM Studio from https://lmstudio.ai/
   * Launch LM Studio and select the model "hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF"
   * Start the local server in LM Studio

4. Configure the server URL:
   * In LM Studio, find the server IP in the API Usage widget
   * Open `mac.py` (for Mac) or `main.py` (for Windows)
   * Update the `LLM_SERVER_URL` variable with your server IP:
     ```python
     LLM_SERVER_URL = "http://YOUR_SERVER_IP:1234/v1/chat/completions"
     ```

5. Choose the appropriate Python file based on your operating system:
   * For Mac: Use `mac.py`
   * For Windows: Use `main.py`

6. Start the application:
   * For Mac:
     ```
     python mac.py
     ```
   * For Windows:
     ```
     python main.py
     ```

7. Open a web browser and navigate to:
   `http://localhost:8000`

## Usage

1. To teach the agent new information:
   * Enter the text in the "Teach" input field.
   * Click "Submit" to store the information.

2. To ask questions about stored knowledge:
   * Enter your question in the "Remember" input field.
   * Click "Submit" to get a response based on the stored information.

## Using Different Models

If you want to use a model other than "hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF":

1. Select your desired model in LM Studio
2. In `mac.py` or `main.py`, update the `MODEL_ID` variable:
   ```python
   MODEL_ID = "Your-Selected-Model-ID"
   ```

## Project Structure

- `mac.py`: Main application file for Mac users
- `main.py`: Main application file for Windows users
- `embedding_storage.py`: Handles storage and retrieval of embeddings
- `index.html`: Frontend interface for the application
- `requirements.txt`: List of Python package dependencies

## Credit

Special thanks to [Maharshi](https://x.com/mrsiipa) for the original idea and pipeline for this project.


## Troubleshooting

If you encounter issues:
* Ensure LM Studio is running and the local server is started
* Verify that the server IP in your Python file matches the one in LM Studio
* Check that all required Python packages are installed
* Look for any error messages in the console

For unresolved issues, please open an issue in the project's repository.

## Contributing

To contribute to the project, fork the repository, make your changes, and submit a pull request.
