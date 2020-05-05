# Meta-Learning Initializations for Image Segmentation

Code for meta-learning and evaluating initializations for image segmentation as described in our paper <https://arxiv.org/abs/1912.06290>.


## Setup


We have included a `requirements.txt` file with dependencies. You can also see `make_python_virtualenv.sh` for recommended steps for setting up your environment.

You can download the FSS-1000 meta-training and evaluation tfrecord shards from:
https://drive.google.com/open?id=1aGHP0ev_1eAFSnYtN0ObDI-DnB0TsQUU


And the joint-training shards from:
https://drive.google.com/open?id=1aQpyQ0CEBCL9EW8xoCaI6xveYxtXNYKq

And the FP-k dataset shards from:
https://drive.google.com/open?id=1G1NJIyQlkxAb4vlsRDPR3W3If_RJ4rPd

We created our meta-training tfrecord shards by following these steps.
Download the FSS-1000 dataset from https://github.com/HKUSTCV/FSS-1000
Convert the images and masks to tfrecords:
```
python fss_1000_image_to_tfrecord.py --input_dir <path to images and masks> --tfrecord_dir <directory to write tfrecords in>
```

## Run the SOTA evaluation

Extract the checkpoint:
```
tar -xzvf EfficientLab-6-3_FOMAML-star_checkpoint.tar.gz
```

Put the FSS-1000 meta-training and evaluation tfrecord shards at the root of this repo or edit the `data_dir` path in `run.sh` to point to the shards on your machine.

Finally, call:
```
./run.sh
```

## Run an experiment

The main point of entry in this codebase is:
```
python run_metasegnet.py <args>
```

See args.py for arguments and their descriptions.

Our SOTA meta-learned initialization that generated the best FSS-1000 results reported in our paper is in this repository at `EfficientLab-6-3_FOMAML-star_checkpoint`

## Visualize predictions
To see predictions, set the environment variable ala:

```
export SAVE_PREDICTIONS=1
```

## EfficientLab
Our SOTA network architecture class is defined in `models/efficientlab.py`.