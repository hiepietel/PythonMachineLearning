from pathlib import Path

scaler = 1

height = 180 / scaler
width = 320 / scaler

height = int(height)
width = int(width)

taug = 10.0
frate = 0.002 #[ms^(-1)]

single_iteration_time = 600
end_spike_time = 200
n = 5 * single_iteration_time + end_spike_time
timeDelta = 0.5  # 5ms
a = 0.02
b = 0.2
d = 8
c = -65

T= n
