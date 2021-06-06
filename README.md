# osu-ml-difficulty

This project aims to assess the difficulty of [osu!](https://osu.ppy.sh) beatmaps using machine learning.


## Aim error

Aim error is the distance between a click and the centre of the target circle. This appears to be normally distributed in two dimensions. 

The model assumes that there is a linear relationship between difficulty of a note and magnitude of error for any particular player. Given an estimate of difficulty, we can use linear regression to estimate player skill parameters. Machine learning algorithms can then be used to give a better estimate of the difficulty of each note. 

## Data

A sample of replays has been collected, available to download [here](https://drive.google.com/drive/u/0/folders/1HiRJbYKJMBZxBa-JJL45xcQuzx0tr_yX). 

It's available as .osr replay files, as well as a reduced size binary format. To convert the replay files to binary format and import to a db, use ./import_replay_data.py

You should also download a dump of all osu maps from [here](https://data.ppy.sh) and set up a [slider](https://llllllllll.github.io/slider/working-with-beatmaps.html#managing-beatmaps-with-a-library) beatmap library, and add the path in config.py

Then `python -m osu_ml_difficulty extract-replay-features` will extract hit error information from the replays.

## Training a model

Once data is set up, `python -m osu_ml_difficulty train` will train the difficulty model

## Evaluating a map

`python -m osu_ml_difficulty map-list <maplist.csv>` will evaluate the difficulty of maps.
The map list should be a csv file containing columns `ID,Mods`

## TODO

- [x] Gather dataset (would be nice to have more top players)
- [x] Parse/extract relevant data
- [x] Implement aim difficulty estimator
- [ ] Tweak and experiment with aim difficulty inputs and model
- [ ] How to handle very hard maps with little/no data?
    - could only do distance corrections like "delta" rework, but that could degrade more time-specific patterns, e.g. alt vs jumps vs spaced streams
    - could multiply delta time a constant amount, so it's actually checking an easier version of each map
    - could scale proportional to distance/time for values outside a given threshold
    - could make sure enough relevant values are fed in as inputs that the NN can effectively extrapolate by itself 
- [ ] Is there bias for notes where you don't aim for the middle? E.g. stacks/streams that encourage hitting the overlapping area 
- [ ] Try aggregating data per map to reduce noise, use average hit error for each note rather than feeding each individual click into the NN.
- [ ] Integrate into ppy/osu for use in pp calculations
- [ ] Is there a nice way to create a model like this for tap?
