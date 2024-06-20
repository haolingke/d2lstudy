import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import convolve, resample

# 示例信号，采样频率 1000 Hz
fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)  # 1秒时间序列
x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)+\
0.5 * np.sin(2 * np.pi * 400 * t)

# 卷积核（低通滤波器示例）
kernel = np.array([0.25, 0.5, 0.25])  # 简单的低通滤波器

# 对信号进行卷积
convolved_signal = convolve(x, kernel, mode='same')

# 降采样：每2个点取一个点，相当于步幅为2
downsampled_signal = convolved_signal[::2]

# 对原始信号、卷积后的信号和降采样后的信号进行傅立叶变换
xf = fft(x)
xf_conv = fft(convolved_signal)
xf_down = fft(downsampled_signal)

# 计算频率轴
N = len(x)
N_down = len(downsampled_signal)
freq = np.fft.fftfreq(N, 1/fs)
freq_down = np.fft.fftfreq(N_down, 1/(fs/2))

# 绘制频域图
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(freq[:N//2], np.abs(xf)[:N//2])
plt.title("Original Signal Frequency Domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(freq[:N//2], np.abs(xf_conv)[:N//2])
plt.title("Convolved Signal Frequency Domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 3)
plt.plot(freq_down[:N_down//2], np.abs(xf_down)[:N_down//2])
plt.title("Downsampled Signal Frequency Domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
