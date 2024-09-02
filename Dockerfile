FROM python:3.11.4
WORKDIR / "C:\Users\91965\Desktop\PROJECTS\docker tutorial"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "./sa.py"]
