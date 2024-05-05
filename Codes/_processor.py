import numpy as np
from scipy import signal

import acconeer.exptool as et


def get_sensor_config():
    config = et.a111.IQServiceConfig()
    config.range_interval = [0.4, 0.8]
    config.update_rate = 60
    config.gain = 0.6
    return config

class ProcessingConfiguration(et.configbase.ProcessingConfig):
    VERSION = 1

    n_dft = et.configbase.FloatParameter(
        label="Estimation window",
        unit="s",
        default_value=5,
        limits=(2, 20),
        updateable=False,
        order=0,
    )

    t_freq_est = et.configbase.FloatParameter(
        label="Time between estimation",
        unit="s",
        default_value=0.15,#0.2
        limits=(0.1, 10),
        updateable=False,
        order=10,
    )

    D = et.configbase.IntParameter(
        label="Distance downsampling",
        default_value=62,
        limits=(0, 248),
        updateable=False,
        order=20,
    )

    f_high = et.configbase.FloatParameter(
        label="Bandpass high freq",
        unit="Hz",
        default_value=float(2.000),
        limits=(0.000, 10.000),
        updateable=False,
        order=30,
    )

    f_low = et.configbase.FloatParameter(
        label="Bandpass low freq",
        unit="Hz",
        default_value=float(0.500),
        limits=(0.000, 10.000),
        updateable=False,
        order=40,
    )

    lambda_p = et.configbase.FloatParameter(
        label="Threshold: Peak to noise ratio",
        default_value=40,
        limits=(1, 1000),
        updateable=False,
        order=50,
    )

    lambda_05 = et.configbase.FloatParameter(
        label="Threshold: Peak to half harmonic ratio",
        default_value=1,
        limits=(0, 10),
        updateable=False,
        order=60,
    )
    
    n_dftBR = et.configbase.FloatParameter(
        label="Estimation window",
        unit="s",
        default_value=15,
        limits=(2, 20),
        updateable=False,
        order=0,
    )

    t_freq_estBR = et.configbase.FloatParameter(
        label="Time between estimation",
        unit="s",
        default_value=0.15,#0.2
        limits=(0.1, 10),
        updateable=False,
        order=10,
    )

    DBR = et.configbase.IntParameter(
        label="Distance downsampling",
        default_value=124,
        limits=(0, 248),
        updateable=False,
        order=20,
    )

    f_highBR = et.configbase.FloatParameter(
        label="Bandpass high freq",
        unit="Hz",
        default_value=0.8,
        limits=(0, 10),
        updateable=False,
        order=30,
    )

    f_lowBR = et.configbase.FloatParameter(
        label="Bandpass low freq",
        unit="Hz",
        default_value=0.2,
        limits=(0, 10),
        updateable=False,
        order=40,
    )

    lambda_Pbr = et.configbase.FloatParameter(
        label="Threshold: Peak to noise ratio",
        default_value=40,
        limits=(1, 1000),
        updateable=False,
        order=50,
    )

    lambda_05BR = et.configbase.FloatParameter(
        label="Threshold: Peak to half harmonic ratio",
        default_value=1,
        limits=(0, 10),
        updateable=False,
        order=60,
    )
class Processor:
    def __init__(self, sensor_config, processing_config, session_info, calibration=None):
        self.config = sensor_config
        # Settings
        # Data length for frequency estimation [s] | 20
        n_dft = processing_config.n_dft
        # Time between frequency estimations [s] | 2
        t_freq_est = processing_config.t_freq_est
        # Time constant low-pass filter on IQ-data [s] | 0.04
        tau_iq = 0.04
        # Time constant low-pass filter on IQ-data [s] | 150
        self.f_s = self.config.update_rate
        # Spatial or Range down sampling factor | 124
        self.D = processing_config.D
        # Lowest frequency of interest [Hz] | 0.1
        self.f_low = processing_config.f_low
        # Highest frequency of interest [Hz] | 1
        self.f_high = processing_config.f_high
        # Time down sampling for DFT | 40 f_s/M ~ 10 Hz
        self.M = int(self.f_s / 10)
        # Threshold: spectral peak to noise ratio [1] | 50
        self.lambda_p = processing_config.lambda_p
        # Threshold: ratio fundamental and half harmonic
        self.lambda_05 = processing_config.lambda_05
        # Interpolation between DFT points
        self.interpolate = True

        self.delta_f = 1 / n_dft
        self.dft_f_vec = np.arange(self.f_low, self.f_high, self.delta_f)
        # self.dft_f_vec = np.around(self.dft_f_vec, decimals=3)
        self.dft_points = np.size(self.dft_f_vec)

        # Butterworth bandpass filter
        f_n = self.f_s / 2
        v_low = self.f_low / f_n
        v_high = self.f_high / f_n
        self.b, self.a = signal.butter(4, [v_low, v_high], btype="bandpass")

        # Exponential lowpass filter
        self.alpha_iq = np.exp(-2 / (self.f_s * tau_iq))
        self.alpha_phi = np.exp(-2 * self.f_low / self.f_s)

        # Parameter init
        self.sweeps_in_block = int(np.ceil(n_dft * self.f_s))
        self.new_sweeps_per_results = int(np.ceil(t_freq_est * self.f_s))
        self.phi_vec = np.zeros((self.sweeps_in_block, 1))
        self.f_est_vec = np.zeros(1)
        self.f_dft_est_vec = np.zeros(1)
        self.snr_vec = 0

        self.sweep_index = 0
        n_dftBR = processing_config.n_dft
        # Time between frequency estimations [s] | 2
        t_freq_estBR = processing_config.t_freq_est
        # Time constant low-pass filter on IQ-data [s] | 0.04
        tau_iqBR = 0.04
        # Time constant low-pass filter on IQ-data [s] | 150
        self.f_sBR = self.config.update_rate
        # Spatial or Range down sampling factor | 124
        self.DBR = processing_config.DBR
        # Lowest frequency of interest [Hz] | 0.1
        self.f_lowBR = processing_config.f_low
        # Highest frequency of interest [Hz] | 1
        self.f_highBR = processing_config.f_high
        # Time down sampling for DFT | 40 f_s/M ~ 10 Hz
        self.MBR = int(self.f_sBR / 10)
        # Threshold: spectral peak to noise ratio [1] | 50
        self.lambda_Pbr = processing_config.lambda_Pbr
        # Threshold: ratio fundamental and half harmonic
        self.lambda_05BR = processing_config.lambda_05BR
        # Interpolation between DFT points
        self.interpolateBR = True

        self.delta_fBR = 1 / n_dftBR
        self.dft_f_vecBR = np.arange(self.f_lowBR, self.f_highBR, self.delta_fBR)
        self.dft_pointsBR = np.size(self.dft_f_vecBR)

        # Butterworth bandpass filter
        f_nBR = self.f_sBR / 2
        v_lowBR = self.f_lowBR / f_nBR
        v_highBR = self.f_highBR / f_nBR
        self.bBR, self.aBR = signal.butter(4, [v_lowBR, v_highBR], btype="bandpass")

        # Exponential lowpass filter
        self.alpha_iqBR = np.exp(-2 / (self.f_sBR * tau_iqBR))
        self.alpha_phiBR = np.exp(-2 * self.f_lowBR / self.f_sBR)

        # Parameter init
        self.sweeps_in_blockBR = int(np.ceil(n_dftBR * self.f_sBR))
        self.new_sweeps_per_resultsBR = int(np.ceil(t_freq_estBR* self.f_sBR))
        self.phi_vecBR = np.zeros((self.sweeps_in_blockBR, 1))
        self.f_est_vecBR = np.zeros(1)
        self.f_dft_est_vecBR = np.zeros(1)
        self.snr_vecBR = 0

        self.sweep_indexBR = 0

    def process(self,data,data_info):
                sweep = data
                sweePbr=data
                if self.sweep_index == 0 or self.sweep_indexBR == 0 :
                    delay_points = int(np.ceil(np.size(sweep) / self.D))
                    self.data_s_d_mat = np.zeros((self.sweeps_in_block, delay_points), dtype="complex")
                    self.data_s_d_mat[self.sweep_index, :] = self.downsample(sweep, self.D)
                    delay_pointsBR = int(np.ceil(np.size(sweePbr) / self.DBR))
                    self.data_s_d_matBR = np.zeros((self.sweeps_in_blockBR, delay_pointsBR), dtype="complex")
                    self.data_s_d_matBR[self.sweep_indexBR, :] = self.downsample(sweePbr, self.DBR)
                    
                    out_data = None
                
                

                elif self.sweep_index < self.sweeps_in_block or self.sweep_indexBR < self.sweeps_in_blockBR :
                    self.data_s_d_mat[self.sweep_index, :] = self.iq_lp_filter_time(
                    self.data_s_d_mat[self.sweep_index - 1, :], self.downsample(sweep, self.D))
                    self.data_s_d_matBR[self.sweep_indexBR, :] = self.iq_lp_filter_timeBR(
                        self.data_s_d_matBR[self.sweep_indexBR - 1, :], self.downsampleBR(sweePbr, self.DBR)
                    )

                    temp_phi = self.unwrap_phase(
                        self.phi_vec[self.sweep_index - 1],
                        self.data_s_d_mat[self.sweep_index, :],
                        self.data_s_d_mat[self.sweep_index - 1, :],
                    )

                    self.phi_vec[self.sweep_index] = self.unwrap_phase(
                        self.phi_vec[self.sweep_index - 1],
                        self.data_s_d_mat[self.sweep_index, :],
                        self.data_s_d_mat[self.sweep_index - 1, :],
                    )

                    phi_filt = signal.lfilter(self.b, self.a, self.phi_vec, axis=0)
                    temp_phiBR = self.unwrap_phaseBR(
                    self.phi_vecBR[self.sweep_indexBR - 1],
                    self.data_s_d_matBR[self.sweep_indexBR, :],
                    self.data_s_d_matBR[self.sweep_indexBR - 1, :],
                        )

                    self.phi_vecBR[self.sweep_indexBR] = self.unwrap_phaseBR(
                        self.phi_vecBR[self.sweep_indexBR - 1],
                        self.data_s_d_matBR[self.sweep_indexBR, :],
                        self.data_s_d_matBR[self.sweep_indexBR - 1, :],
                    )

                    phi_filtBR = signal.lfilter(self.bBR, self.aBR, self.phi_vecBR, axis=0)
                    out_data = {
                        "phi_raw": self.phi_vec,
                        "phi_filt": phi_filt,
                        "power_spectrum": np.zeros(self.dft_points),
                        "x_dft": np.linspace(self.f_low, self.f_high, self.dft_points),
                        "f_dft_est_hist": self.f_dft_est_vec,
                        "f_est_hist": self.f_est_vec,
                        "f_dft_est": 0.000,
                        "f_est": 0.0000,
                        "f_low": self.f_low,
                        "f_high": self.f_high,
                        "snr": 0,
                        "lambda_p": self.lambda_p,
                        "lambda_05": self.lambda_05,
                        "dist_range": self.config.range_interval,
                        "init_progress": round(100 * self.sweep_index / self.sweeps_in_block),
                        "phi_rawBR": self.phi_vecBR,
                        "phi_filtBR": phi_filtBR,
                        "power_spectrumBR": np.zeros(self.dft_pointsBR),
                        "x_dftBR": np.linspace(self.f_lowBR, self.f_highBR, self.dft_pointsBR),
                        "f_dft_est_histBR": self.f_dft_est_vecBR,
                        "f_est_histBR": self.f_est_vecBR,
                        "f_dft_estBR": 0,
                        "f_estBR": 0,
                        "f_lowBR": self.f_lowBR,
                        "f_highBR": self.f_highBR,
                        "snrBR": 0,
                        "lambda_Pbr": self.lambda_Pbr,
                        "lambda_05BR": self.lambda_05BR,
                        "dist_rangeBR": self.config.range_interval,
                        "init_progressBR": round(100 * self.sweep_indexBR / self.sweeps_in_blockBR),
                    }
                else:
        # Lowpass filter IQ data downsampled in distance points
                    self.data_s_d_mat = np.roll(self.data_s_d_mat, -1, axis=0)
                    self.data_s_d_mat[-1, :] = self.iq_lp_filter_time(
                        self.data_s_d_mat[-2, :], self.downsample(sweep, self.D)
                    )
                    self.data_s_d_matBR = np.roll(self.data_s_d_matBR, -1, axis=0)
                    self.data_s_d_matBR[-1, :] = self.iq_lp_filter_time(
                            self.data_s_d_matBR[-2, :], self.downsample(sweePbr, self.DBR)
                    )

        # Phase unwrapping of IQ data
                    temp_phi = self.unwrap_phase(
                            self.phi_vec[-1], self.data_s_d_mat[-1, :], self.data_s_d_mat[-2, :]
                     )
                    self.phi_vec = np.roll(self.phi_vec, -1, axis=0)
                    self.phi_vec[-1] = temp_phi
                    temp_phiBR = self.unwrap_phase(
                    self.phi_vecBR[-1], self.data_s_d_matBR[-1, :], self.data_s_d_matBR[-2, :]
                    )
                    self.phi_vecBR = np.roll(self.phi_vecBR, -1, axis=0)
                    self.phi_vecBR[-1] = temp_phiBR
                    if np.mod(self.sweep_index, self.new_sweeps_per_results - 1) == 0:
            # Bandpass filter unwrapped data
                            phi_filt_vec = signal.lfilter(self.b, self.a, self.phi_vec, axis=0)
                            P, dft_est, _ = self.dft(self.downsample(phi_filt_vec, self.M))
                            f_heart_est, _, snr, _ = self.heart_rate_est(P)  # Modified line

                            self.f_est_vec = np.append(self.f_est_vec, f_heart_est)  # Modified line
                            self.f_dft_est_vec = np.append(self.f_dft_est_vec, dft_est)
                            self.snr_vec = np.append(self.snr_vec, snr)

                    if np.mod(self.sweep_indexBR, self.new_sweeps_per_resultsBR - 1) == 0:
                # Bandpass filter unwrapped data
                            phi_filt_vecBR = signal.lfilter(self.bBR, self.aBR, self.phi_vecBR, axis=0)
                            Pbr, dft_estBR, _ = self.dft(self.downsample(phi_filt_vecBR, self.MBR))
                            f_breath_estBR, _, snrBR = self.breath_freq_estBR(Pbr)

                            self.f_est_vecBR = np.append(self.f_est_vecBR, f_breath_estBR)
                            self.f_dft_est_vecBR = np.append(self.f_dft_est_vecBR, dft_estBR)
                            self.snr_vecBR = np.append(self.snr_vecBR, snrBR)

                            out_data = {
                                    "phi_raw": self.phi_vec,
                                    "phi_filt": phi_filt_vec,
                                    "power_spectrum": P,
                                    "x_dft": np.linspace(self.f_low, self.f_high, self.dft_points),
                                    "f_dft_est_hist": self.f_dft_est_vec,
                                    "f_est_hist": self.f_est_vec,
                                    "f_dft_est": dft_est,
                                    "f_est": f_heart_est,  # Modified line
                                    "f_low": self.f_low,
                                    "f_high": self.f_high,
                                    "snr": snr,
                                    "lambda_p": self.lambda_p,
                                    "lambda_05": self.lambda_05,
                                    "dist_range": self.config.range_interval,
                                    "init_progress": None,
                                    "phi_rawBR": self.phi_vecBR,
                                    "phi_filtBR": phi_filt_vecBR,
                                    "power_spectrumBR": Pbr,
                                    "x_dftBR": np.linspace(self.f_lowBR, self.f_highBR, self.dft_pointsBR),
                                    "f_dft_est_histBR": self.f_dft_est_vecBR,
                                    "f_est_histBR": self.f_est_vecBR,
                                    "f_dft_estBR": dft_estBR,
                                    "f_estBR": f_breath_estBR,
                                    "f_lowBR": self.f_lowBR,
                                    "f_highBR": self.f_highBR,
                                    "snrBR": snrBR,
                                    "lambda_Pbr": self.lambda_Pbr,
                                    "lambda_05BR": self.lambda_05BR,
                                    "dist_rangeBR": self.config.range_interval,
                                    "init_progressBR": None,
                                }
                    else:
                                out_data = None

                self.sweep_index += 1
                self.sweep_indexBR += 1
                return out_data
    def downsample(self, data, n):
            return data[::n]
    def iq_lp_filter_time(self, state, new_data):
            return self.alpha_iq * state + (1 - self.alpha_iq) * new_data
    def unwrap_phase(self, phase_lp, data_1, data_2):
            return phase_lp * self.alpha_phi + np.angle(np.mean(data_2 * np.conjugate(data_1)))
                
                
    def dft(self, data):
            data = np.squeeze(data)
            n_vec = np.arange(data.size) * self.M
            dft = np.exp((2j * np.pi / self.f_s) * np.outer(self.dft_f_vec, n_vec))
            P = np.square(np.abs(np.matmul(dft, data)))
        
        # Check if the peak index is within the valid range
            if np.argmax(P) >= len(self.dft_f_vec):
                dft_est = 0
            else:
                dft_est = self.dft_f_vec[np.argmax(P)]

            dft_est = np.round(dft_est, 4)
            #print(dft_est)

            return P, dft_est, P[np.argmax(P)]

    def noise_est(self, P):
            return np.mean(np.sort(P)[: (self.dft_points // 2) - 1])
   
    def half_peak_frequency(self, P, f_est):
            idx_half = int(f_est / (2 * self.delta_f))
            if idx_half < self.f_low:
                return 0
            else:
                return (1 / self.delta_f) * (
                    (self.dft_f_vec[idx_half + 1] - f_est / 2) * P[idx_half]
                    + (f_est / 2 - self.dft_f_vec[idx_half]) * P[idx_half + 1]
                )
        
    
    def downsampleBR(self, data, n):
                return data[::n]

    def iq_lp_filter_timeBR(self, stateBR, new_dataBR):
         return self.alpha_iqBR * stateBR + (1 - self.alpha_iqBR) * new_dataBR

    def unwrap_phaseBR(self, phase_lPbr, data_1BR, data_2BR):
            return phase_lPbr * self.alpha_phiBR + np.angle(np.mean(data_2BR * np.conjugate(data_1BR)))

    def dftBR(self, data):
            dataBR = np.squeeze(data)
            n_vecBR = np.arange(dataBR.size) * self.MBR
            dftBR = np.exp((2j * np.pi / self.f_sBR) * np.outer(self.dft_f_vecBR, n_vecBR))
            Pbr = np.square(np.abs(np.matmul(dftBR, dataBR)))
            idx_fBR = np.argmax(Pbr)
            dft_estBR = self.dft_f_vecBR[idx_fBR]
            return Pbr, dft_estBR, Pbr[idx_fBR]

    def noise_estBR(self, Pbr):
            return np.mean(np.sort(Pbr)[: (self.dft_pointsBR // 2) - 1])

    def half_peak_frequencyBR(self, Pbr, f_estBR):
            idx_halfBR = int(f_estBR / (2 * self.delta_fBR))
            if idx_halfBR < self.f_lowBR:
                return 0
            else:
                return (1 / self.delta_fBR) * (
                (self.dft_f_vecBR[idx_halfBR + 1] - f_estBR / 2) * Pbr[idx_halfBR]
                + (f_estBR / 2 - self.dft_f_vecBR[idx_halfBR]) * Pbr[idx_halfBR + 1]
                )

    def breath_freq_estBR(self, Pbr):
            f_idxBR = np.argmax(Pbr)
            P_peakBR = Pbr[f_idxBR]

            if self.interpolateBR:
                f_estBR, P_peakBR = self.freq_quad_interpolationBR(Pbr)
            else:
                f_estBR = self.dft_f_vecBR[f_idxBR]

            P_halfBR = self.half_peak_frequencyBR(Pbr, f_estBR)

            if P_peakBR < self.lambda_05BR * P_halfBR:
             f_estBR = f_estBR/ 2
             P_peakBR = P_halfBR
            #  print(f_estBR)
             if(f_estBR>0.2 and f_estBR<0.8):
              f_est_validBR = True
            else:
              f_est_validBR=False
            # if self.f_lowBR < f_estBR < self.f_highBR and P_peakBR > self.lambda_Pbr * self.noise_estBR(Pbr):
            #     f_est_validBR = True
            # else:
            #     f_est_validBR = False
            #     f_estBR = 0

            snrBR = P_peakBR / self.noise_estBR(Pbr)
            
            return f_estBR, P_peakBR, snrBR

    def freq_quad_interpolationBR(self, Pbr):
            f_idxBR = np.argmax(Pbr)

            if 0 < f_idxBR < (Pbr.size - 1) and Pbr.size > 3:
                f_estBR = self.dft_f_vecBR[f_idxBR] + self.delta_fBR / 2 * (
                (np.log(Pbr[f_idxBR + 1]) - np.log(Pbr[f_idxBR - 1]))
                / (2 * np.log(Pbr[f_idxBR]) - np.log(Pbr[f_idxBR + 1]) - np.log(Pbr[f_idxBR - 1]))
                )
                P_peakBR = Pbr[f_idxBR] + np.exp(
                (1 / 8)
                * np.square(np.log(Pbr[f_idxBR + 1]) - np.log(Pbr[f_idxBR - 1]))
                / (2 * np.log(Pbr[f_idxBR]) - np.log(Pbr[f_idxBR + 1]) - np.log(Pbr[f_idxBR - 1]))
                )

                if not (self.f_lowBR < f_estBR < self.f_highBR):
                    f_estBR = 0
            else:
                f_estBR = 0
                P_peakBR = 0

            return f_estBR, P_peakBR
    def heart_rate_est(self, P):
            f_idx = np.argmax(P)
            P_peak = P[f_idx]

            if self.interpolate:
                f_est, P_peak = self.heart_rate_quad_interpolation(P)
            else:
                f_est = self.dft_f_vec[f_idx]

            P_half = self.half_peak_frequency(P, f_est)

            if P_peak < self.lambda_05 * P_half:
                f_est = f_est / 2
                P_peak = P_half

            if self.f_low < f_est < self.f_high and P_peak > self.lambda_p * self.noise_est(P):
                f_est_valid = True
            else:
                f_est_valid = False
                f_est = 0

            snr = P_peak / self.noise_est(P)
            return f_est, P_peak, snr, f_est_valid
   
       
     
    
   
    def heart_rate_quad_interpolation(self, P):
            f_idx = np.argmax(P)

            if 0 < f_idx < (P.size - 1) and P.size > 3:
                f_est = self.dft_f_vec[f_idx] * self.f_s + self.delta_f / 2 * (
                (np.log(P[f_idx + 1]) - np.log(P[f_idx - 1]))
                / (2 * np.log(P[f_idx]) - np.log(P[f_idx + 1]) - np.log(P[f_idx - 1]))
                )
                P_peak = P[f_idx] + np.exp(
                (1 / 8)
                * np.square(np.log(P[f_idx + 1]) - np.log(P[f_idx - 1]))
             / (2 * np.log(P[f_idx]) - np.log(P[f_idx + 1]) - np.log(P[f_idx - 1]))
             )

                if not (self.f_low < f_est < self.f_high):
                    f_est = 0
            else:
                    f_est = 0
                    P_peak = 0

            return f_est, P_peak