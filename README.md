# Emotion dataset for NLP

## How to intsall

Create a new conda environment and install the required packages
```shell
conda create --name <ENV_NAME> --file requirements.txt
```
Activate the environment and install spacy English anguage model
```shell
conda activate <ENV_NAME>
python -m spacy download en
```

## How to run demo

simply type ```streamlit run demo.py``` on terminal in the project 
main folder, and a new webpage will open with the demo.