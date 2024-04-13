# Apps using Langchain and ChatGPT

Apps created as part of a
[course](https://www.udemy.com/course/chatgpt-and-langchain-the-complete-developers-masterclass/)
on langchain and chatGPT from Udemy.

## Setup Requirements

- Python 3.11+
- Pipenv
- API key from OpenAI
- VScode IDE + Python extensions (for development)

## How to run

- Each app maintains it's own dependencies in a `Pipfile`
- Each app has it's own `.env` file that must be created following the
  `.env.example` file in the app's directory
- Before running the app, install dependencies and create the `.env` file
- Activate the virtual environment for the app with `pipenv shell` before
  running the app
- After running the app, remember to switch the virtual env when running another app
- The app 'langchain-pdf' has it's own README with additional instructions to setup and run

## Useful pipenv commands

- `pipenv install` - Installs dependencies
- `pipenv shell` - Activates virtual environment
 