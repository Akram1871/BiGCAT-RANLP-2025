# BiGCAT-for-ECIR24

Install Dependencies and Create an Environment
```python
pip install -r requirements.txt
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
pip install torch-geometric
```

Now run the following command:
Here exwcute, "run3.py" for FiNER-ORD dataset and "run4.py" for FIN dataset
```python
python run3.py --model_name=philschmid/distilroberta-base-ner-conll2003 --epoch=25 --batch=16 --lr=2.3e-5 --max_span=8 --emb_width=128 --project_dim=256
```
