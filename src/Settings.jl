module Settings
using Parameters
export Params

@with_kw struct Params

# Lorenz96 parameters
N = 10						# number of dimensions >= 4
F = 8.0						# input driving dynamics parameter
dt = 0.01					# sampling time
T = 100.0					# final time
warmup = 10.0				# time to skip at start to avoid initial startup period
fixed_init = false			# initial conditions: true => fixed, false => random

# define ESN parameters, Lorenz here refers to ReservoirComputing docs
res_size = 20			# Lorenz example uses 300
res_radius = 1.2		# default is 1.0, Lorenz example uses 1.2
res_sparsity = 0.3		# default is 0.1, Lorenz example uses 0.02
# ave num res nodes directly influence by input is res_size * input_sparsity
input_scaling = 0.1		# default is 0.1
input_sparsity = 0.15	# default is 0.1, if 1.0, then inputs goes to all res nodes	

# fitting
shift = 0.5				# goal to predict first trajectory dim shift units into future
train_frac = 0.7		# split of data into training and testing

# convenience items
run_num = 1				# store run number within experiment for filename saving

end	#struct

end # module Settings
