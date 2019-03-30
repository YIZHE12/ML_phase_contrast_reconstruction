# phase_contrast_reconstruction

X-ray phase contrast image can provided much higher contrast among soft tissues, therefore, more ideal for breast cancer diagnosis. However, to extract the phase information from the X-ray image, additional experimental setup and signal recovery method is required.
The raw data is extremely noise and required sophiscated mathmatrical method to extract useful signal.
T
he pair of raw data for signal extraction:

<img src=raw_data_I.jpg height = 300> <img src=raw_sandpaper.jpg height = 300>
Raw data can be downloaded in https://sciencedata.dk/shared/breast_speckle


Here, we demonstrated three different methods to extract the phase information based on 'speckle tracking X-ray phase contrast image' of a breast tumour. As this is simulation data, we are able to know the ground truth values to compared with the extracted phase values from the raw data. 

<img src=Github_phasecontrast.jpg height = 300>

Method 1: [Digital image cross correlation](phase_contrast_reconstruction/phase_contrast_reconstruction/Cross_cor_main.py)

Method 2: [Unified Modulated Pattern Analysis](https://github.com/pierrethibault/UMPA)

Method 3: [Use turncated newton method to minimize the cost function](phase_contrast_reconstruction/Iterative_cal.py)
