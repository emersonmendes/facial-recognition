# facial-recognition

## Dev

``` bash
# Install venv
python3 -m venv ./env

# Activate env
source ./env/bin/activate

# Install deps
pip install -r requirements.txt

# Update requirements.txt with current deps
pip freeze > requirements.txt
```

## Vscode

``` json

{
    "python.pythonPath": "${workspaceFolder}/env/bin/python",
    "python.venvPath": "${workspaceFolder}/env",
    "python.linting.pylintArgs" : ["--generate-members"]
}

```
