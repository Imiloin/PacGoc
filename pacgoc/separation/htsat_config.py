loss_type = "clip_bce"

resume_checkpoint = None

# for model's design
enable_tscam = True # enbale the token-semantic layer

# for signal processing
sample_rate = 16000 # 16000 for scv2, 32000 for audioset and esc-50
window_size = 1024
hop_size = 320 # 160 for scv2, 320 for audioset and esc-50
mel_bins = 64
fmin = 50
fmax = 14000
# shift_max = int(clip_samples * 0.5)

# for data collection
classes_num = 527 # esc: 50 | audioset: 527 | scv2: 35

# for htsat hyperparamater
htsat_window_size = 8
htsat_spec_size =  256
htsat_patch_size = 4 
htsat_stride = (4, 4)
htsat_num_head = [4,8,16,32]
htsat_dim = 256 # for 2048-d model
htsat_depth = [2,2,6,2]
