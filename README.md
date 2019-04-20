# hair-dye

## Create environment

```
$ conda env create -f environment.yml
$ source activate hairdye
```

when you done all the training things, at last you deactivate environment by running `source deactivate`.
And the next time you want to train your network, just run `source activate hairdye`, you only have to create the environment once.

## Download dataset

```
$ sh download.sh
```

## Train
```
nohup python -u main.py --mode=train > out.log &
```

The checkpoint file and sample images can be seen in src/checkpoint/default/ directory.
