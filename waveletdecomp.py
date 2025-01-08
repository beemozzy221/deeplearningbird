from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import pywt

#Extract information from the wav file
def waveletsegment(wav_fname):
    #Extract information
    samplerate, data = wavfile.read(wav_fname)
    print(f"shape = {data.shape}")

    #Stereo ignorance
    if len(data.shape) > 1:
        print(f"number of channels = {data.shape[1]}")
        data = data[:, 0]

    #Normalize the data
    data = data / np.max(np.abs(data))

    #Obtain length
    length = data.shape[0] // samplerate
    print(f"length = {length}s")

    return segmentwavfile(data, length)

# Perform Wavelet Packet Decomposition (WPD)
def wavpacketdecomp(data):
    return pywt.WaveletPacket(data, wavelet='dmey', mode='symmetric', maxlevel=5)

# Collect coefficients from each node
def collect_coefficients(wpt):
    nodes = wpt.get_level(wpt.maxlevel, order="natural")
    coefficients = {node.path: node.data for node in nodes}
    coefficients_array = [node.data for node in nodes]
    energy_array = [np.sum(np.square(node.data)) for node in nodes]


    return coefficients, coefficients_array, energy_array

#Pad arrays
def padding(dataarray, length):
    padding_size = length - dataarray.shape[0] % length
    padded_array = np.pad(dataarray,(0,padding_size),mode = "constant")

    return padded_array.reshape(length, -1)

#Segment the wav files into 1 seconds
def segmentwavfile(data, length):
    return padding(data, length)

#Plot the Amplitude vs Time graph
def plot (data, length):
    time  = np.linspace(0,length, data.shape[0])
    plt.plot(time, data[:,0], label = "left channel")
    plt.plot(time, data[:,1], label = "right channel")
    plt.xlabel('Time')
    plt.ylabel("Amplitude")
    plt.show()

def coefficients(wpt):
    return collect_coefficients(wpt)

# Visualize coefficients
plt.figure(figsize=(12, 8))

def energyinfowavelets(coefficients):
    energy = {path: np.sum(np.square(coeff)) for path, coeff in coefficients.items()}
    plt.bar(energy.keys(), energy.values())
    plt.xlabel("Node path")
    plt.ylabel("Energy")
    plt.xticks(rotation=90, fontsize=8)
    plt.show()

#Print all coefficients of the wavelets
def plotprintwavelets(coefficients):
    for i, (path, coeff) in enumerate(coefficients.items()):
        plt.subplot(8, 8, i + 1)
        plt.plot(coeff, lw=0.5)
        plt.title(path, fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Save coefficients to a file (optional)
def savecoeinnumpyz(coefficients):
    np.savez("wavelet_coefficients.npz", **coefficients)
    print("Wavelet packets saved.")

def savenumpy(file_path, data):
    np.save(file_path, data)
    print(f"Saved successfully at {file_path}!")












