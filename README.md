## Python Template Repository

Setup with useful VSC, Git configuration and libraries. 

To start:
```
    ProjectRoot>python -m venv .
    ProjectRoot>.\Scipts\activate
    ProjectRoot>pip install -r requirements.txt
```

To test:
```
    ProjectRoot>pytest
```

To generate coverage documentation:
```
    ProjectRoot>coverage run -m pytest
```

To view coverage report as a HTML document:
```
    ProjectRoot>coverage html
```

Remember when revisiting after the IDE or terminal has been restarted, restart the env;

<sub>On Linux/Mac this will be `./bin/activate`, on M1 Mac use `source ./bin/activate.csh` from a csh terminal</sub>
```
    ProjectRoot>.\Scipts\activate
```
When the env is active you should see the project name before the directory in ther terminal;
```
    (ProjectName) ProjectName>
```

To update the dependancy list, use;
```
    ProjectName>pip install -r requirements.txt 
```
To install dependancies, use;
```
    ProjectRoot>pip install -r requirements.txt
```
