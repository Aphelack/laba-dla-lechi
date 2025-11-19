```bash
git clone https://github.com/Aphelack/laba-dla-lechi.git
cd laba-dla-lechi
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download ru_core_news_sm
python generate_dataset.py
python train_model.py
python predict.py "Есть ли льготные билеты на концерт для студентов?"
```