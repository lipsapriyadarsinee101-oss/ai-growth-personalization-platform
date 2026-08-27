.PHONY: setup data test app api docker
setup:
	python -m pip install -r requirements.txt
data:
	python scripts/generate_data.py
test:
	pytest -q
app:
	streamlit run app.py
api:
	uvicorn api.main:app --reload
docker:
	docker build -t ai-growth-platform .

