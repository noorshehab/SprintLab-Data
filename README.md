# Conventions
Tests -> Test folder.
Requirements -> separate requirements in separate .txt files in the requirements folder to make installing dependencies for testing each module easier.
Notebooks/Experiments -> where trying machine learning ideas will happen as well as any EDA and its outputs.
ML-> where training the fina models will be done and where the services wrapping model calls will live.
API-> Django app.
Workflows -> workflow for push/pr to each branch will automate running the relevant test scripts.
## Actions Template
### Example .yml file
name: Test [Name of Module/Script]

``` yml
run-name: ${{ github.actor }} is testing [Name of Module/Script] 
on: 
  push:
    paths:
      - '[Module subfolder]/**' // The /** makes it so this action runs whenever a file in the folder is updated
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements/[Name of Module/Script].txt 
      - run: pytest tests/test_[Name of Module/Script].py 
      
``` 
