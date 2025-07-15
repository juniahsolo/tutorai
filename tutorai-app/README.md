# Language Tutor App

This project is a Streamlit application that allows users to upload an audio file, transcribe the audio using Whisper, generate a text response using a language model, and synthesize and play an audio response using TTS.

## Project Structure

```
tutorai-app
├── app.py               # Main logic for the Streamlit application
├── requirements.txt     # Lists the dependencies required for the project
└── README.md            # Documentation for the project
```

## Installation

To run this application, you need to have Python installed on your machine. Follow these steps to set up the project:

1. Clone the repository or download the project files.
2. Navigate to the project directory:
   ```
   cd tutorai-app
   ```
3. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

To start the Streamlit application, run the following command in your terminal:
```
streamlit run app.py
```

This will launch the application in your default web browser.

## Usage

1. Upload an audio file in MP3 or WAV format.
2. The application will transcribe the audio using Whisper.
3. A text response will be generated using a language model.
4. The response will be synthesized into audio and played back to you.

## Dependencies

The project requires the following Python packages:

- Streamlit
- Whisper
- Transformers
- TTS
- Torch
- Torchaudio

Make sure all dependencies are installed before running the application.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.