# Asynchronous Advantage Actor with Centralized Critic and Communication (A3C3)

## Reference paper

In this project, I wish to implement the A3C3 paper:

- **A3C2** -- the older version
  - David Simoes, Nuno Lau, Paulo Reis, [Asynchronous Advantage Actor-Critic with Communication (A3C2) page link](https://sciendo.com/article/10.2478/jaiscr-2020-0013)

  - A3C2 github page [here](https://github.com/david-simoes-93/A3C2/tree/master)

- **A3C3** -- the newer version
  - David Simoes, Nuno Lau, Paulo Reis, [Asynchronous Advantage Actor with Centralized Critic and Communication (A3C3) page link](https://www.sciencedirect.com/science/article/pii/S0925231220301314)

  - [link to the A3C3 web-based paper](https://www.sciencedirect.com/science/article/pii/S0925231220301314/pdfft?md5=dfa0ac751c44da64210b356a6d1b24e9&pid=1-s2.0-S0925231220301314-main.pdf)

  - **A3C3** github page [here](https://github.com/david-simoes-93/A3C3/tree/master)

## Virtual Environment Setup

This creates a virtual environment for the project for a specific python version, in this case `python 3.11.1`. Note that it assumes you have already downloaded and installed the version you want to use.

```python
C:\Users\yusup\AppData\Local\Programs\Python\Python36\python.exe -m venv a3c3_311
```

To activate the `virtual environment`:

```python
.path_to_virtual/Scripts/activate
```

For e.g.,

```python
.\a3c3_311\Scripts\activate
```

To deactivate it:

```python
deactivate
```

## Install libraries

All the libraries and dependencies used throughout in this project are listed in the `requirement.txt` file.
To `install`:

```bash
pip3 install -r requirements.txt
```

## Lauch Tensorboard

While training is taking place, statistics on agent(s) performance(s) is/are available from Tensorboard. For only worker 0:

```bash
tensorboard --logdir=tb_logs/worker_0:'./train_0' --host=127.0.0.1
```

For workers 0, 1, and 2:

```bash
tensorboard --logdir=tb_logs/worker_0:'./train_0',tb_logs/worker_1:'./train_1',tb_logs/worker_2:'./train_2' --host=127.0.0.1
```

For all the workers:

```bash
tensorboard --logdir=tb_logs/ --host=127.0.0.1
```

Or

```bash
tensorboard --logdir=tb_logs/
```

Then you can launch it with `localhost:6006` if you didn't specify the port.

## Problems and Solutions

### Carriage Return Characters

If you run the `.sh` files (on Linux/Unix) and encountered an error about `\r`, carriage-return character. It means that the file was edited on a DOS or a Windows based system, and that Linux/Unix does not like the `\r`.

To solve the problem, you can use `dos2unix` to convert the carriage return line endings to the correct format. However, the easier way is to use `awk` by copy pasting the snippet below into a bash terminal, where `NavBatch` is the file which contains the `\r`.

```bash
awk '{ sub("\r$", ""); print }' NavBatch.sh > NavBatch2.sh
mv NavBatch2.sh NavBatch.sh
```

### Remove the 260 Character Path Limit by Editing the Registry

If you face an error similar to this:

```python
ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory: 'g:\\my drive\\ai_robotics\\y1-s1-reinforcement-learning\\project\\a3c3_git\\a3c3\\a3c3_36\\Lib\\site-packages\\numpy\\compat\\__init__.py'
```

Windows 10 has a limit on the length of characters on a (file, folder) path. The max character length is 260 characters. To remove/extend the limit, see [this](https://www.howtogeek.com/266621/how-to-make-windows-10-accept-file-paths-over-260-characters/)
