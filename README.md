# wxgov-custom-metric

## Table of Contents

- Developing locally
  - [Native runtime](#native-runtime)
  - [Containerized](#containerized)

## Native runtime

To run this code in your computer execute the following commands into project root directory

```bash
cd <directory to store your Python environment>
python -m venv <your-venv-name>

source <your-venv-name>/bin/activate

$ pip install --no-cache-dir -r requirements.txt

$ python main.py
```

#### Dectivate your Python virtual environment

If you need to change to a different environment, you can deactivate your current environment using the command below:

```bash
deactivate
```

## Containerized

To run this code using Docker container execute the following commands into project root directory

```bash
$ docker build --platform linux/amd64 -t <image_name> .
$ docker run -p 8000:8000 <image_name>
```
