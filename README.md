# hair-dye

## Create environment

```
$ conda env create -f environment.yml
$ source activate hairdye
```

After you have done all the training and testing things, in the end, you can deactivate the environment by running `source deactivate`.
And the next time you want to train your network, just run `source activate hairdye`, you only have to create the environment once.

## Download dataset

```
$ sh download.sh
```

## Train

```
$ nohup python -u main.py --mode=train > out.log &
```

The checkpoint file and sample images can be seen in `src/checkpoint/default/` directory.

## Test
```
$ python main.py --mode=test
```

## Run

Plot a groundtruth image and the predicted segmentation

```
python  main.py --mode=run --set=test --image=27
```

`set` can be one `train` and `test`, default is `test`
`image` is the image index of the set, default is `0`
