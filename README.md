# Promote mental health and productivity at work using deep learning

This prototype was build as part of the Startup Weekend 2022 at the Apex Center for Entrepreneurship at Virginia Tech. Our team was among the three winning teams.

The idea was to build an application that monitors the real time emotional state of our users in order to improve well-being and eliminate burnout, which in turn improves productivity and employee retention.
We accomplish this using state-of-the-art machine learning and AI to analyze a user’s mood through their facial expressions.

For this prototype, we use a pre-trained emotion detection model and notify people if they are getting tired and need a break in order to stay productive.
We also send a notification, if people are having a great time.

## Setup guide

* install `python` and `pip`
* clone the repository
* run `pip install -r requirements.txt` to install the required packages
* to start the monitoring use:

```bash 
cd src
python emotions.py
```

## Future work

We started to implement a server using flask (see `flask_server.py`). This can be used as a starting point to provide more advanced statistics on a webpage.

Apart from that, sharpening the problem and exploring existing solutions are good next steps before continuing to further develop this prototype.

## References

This project is heavily based on the "Emotion-Detection" project from @atulapra ([link to repo](https://github.com/atulapra/Emotion-detection))