# Acoustic Localization

An attempt at 2D localization using a microphone array.
The primary purpose will be to compare several estimators in order to evaluate their performance (accuracy, uncertainty, confidence, etc).
Hopefully we will have a camera tracking solution implemented as well to provide a more accurate reference trajectory.
## Installation

Run 'pip2 install -r requirements.txt' for Python 2, or 'pip3 install -r requirements.txt' for Python 3 to install dependencies.

## Approach

### **Option A:** Amplitude

Using a fixed frequency, we can use the amplitude of the signal reaching each microphone to estimate the distance.
Using two known distances is the bare minimum for generating a solution. With 3 microphones, we can hopefully provide a more robust position estimate.
Using a known frequency will allow filtering of noise for better response. Depending on the total uncertainty, we can play two audio signals simultaneously to determine heading of the robot.
Total Uncertainty will be determined by the calibration/response functions gathered experimentally.
Depending on the uncertainty of the amplitude<->distance function, and the amplitude<->angle function, we will know how accurate the system can be.

### **Option B:** Angry Bird

Impulse signals (low to moderate frequency chirping or clicking) can be easily detected and used to determine the arrival time of the signal to each microphone.
It takes 3 microphones minimum to solve this version of the problem, which means we will need microphones other than the webcams, but thankfully no calibration is needed for this option.
This method has the potential to be more accurate if we can synchronize the audio signals well.
There is approximately 3ms of delay per meter of difference.
Even if the audio signal is not well time synchronized, the delay can be recovered by the estimator.

## **Implementation**

All processing will happen offline, after a semi-large dataset has been gathered. The goal is to capture time-synchronized audio from each of the microphones.

### Data Collection

We will be using 3x Logitech C922 Webcams.
Recording audio from these should be achievable because they are USB webcams, and I think python's audio library can record from an arbitrary number of devices at the same time.
The 2D transform between each microphone should be recorded for use in estimation.
We will also record a video feed for camera-based localization later.
The issue will be audio synchronization.

#### Synchronization

Synchronizing the signals can be done in post as long as we tap the microphones to one another in pairs.
(tapping the mics together should give an instantaneous impulse to both, which can be used to align the time-series)
Also the time-delay between microphones can also be a parameter that the estimator tries to estimate.

### Estimator functions

- [ ] decide on format for probability functions
  - symbolic? python has sympy, and there's always matlab.
  - discrete, like a lookup-table?
  - function-based/black box & standardize the return format?

## Prereqs

### Microphone Calibration

#### distance vs amplitude measurements

The expected response function is ```d ~ sqrt(1/A)```. Experimental data will give form of relationship, but also an idea of the variance as a function of amplitude, and a probability distribution, since we assume noise is gaussian.

#### angle vs amplitude measurements

This relationship is mostly unknown (and may not have a pretty analytical form).  ```G = f(\theta)``` in general is expected \theta.

### The Robot

Need a software controllable robot which can have a speaker mounted.
Exact robot TBD. But it would be better if it were nonholonomic drive so that it has a model that's worth

## Stretch Goals

- implement 2D localization for the same robot for comparison/ground truth
- turn into real-time audio

