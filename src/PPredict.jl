module PPredict
using ReservoirComputing, DynamicalSystems, Plots
include("Settings.jl")
using .Settings
include("Machines.jl")
using .Machines
export Params
export best
export predict_driver

# see https://en.wikipedia.org/wiki/Lorenz_96_model
# The Lorenz-96 model is predefined in DynamicalSystems.jl
# add initial warmup period then cut it off
# add final extra time for shift data, process it later
# trajectory is a Dataset, convert to matrix
function lorenz96(S::Params; fixed_init=S.fixed_init)
	if fixed_init
		u = S.F * ones(S.N)
		u[1] += 0.01					# small perturbation 
	else
		u = S.F * rand(S.N)
	end
	ds = Systems.lorenz96(S.N, u; F = S.F)
	trj = Matrix(trajectory(ds, S.T+S.warmup+S.shift, u; Δt = S.dt))
	first_ind = Int(round(S.warmup/S.dt)+1)
	return trj[first_ind:end,:]
end

make_esn(S, input_data) = ESN(input_data;
    	reservoir = RandSparseReservoir(S.res_size, radius=S.res_radius, 
    					sparsity=S.res_sparsity),
    	input_layer = SparseLayer(scaling=S.input_scaling, sparsity=S.input_sparsity),
    	reservoir_driver = GRU())

# take existing esn and generate new states for new input
new_states(input_data, esn) = ESN(input_data;
		reservoir=esn.reservoir_matrix,
		input_layer=esn.input_matrix,
		reservoir_driver=esn.reservoir_driver).states

# input includes all dimensions, target is first dimension shifted by S.shift
# esn requires data w/ features in rows and time in columns, so must transpose return
function make_data(S)
	input_data = lorenz96(S);
	shift_steps = Int(round(S.shift/S.dt))
	len = size(input_data)[1] - shift_steps
	@assert shift_steps < len
	# target data is first input shifted to right by shift_steps
	target_data = input_data[1+shift_steps:end,1]
	# slice final shift_steps off of input
	input_data = input_data[1:len,:]
	return transpose(input_data), target_data
end

function predict_driver(S)
	input_data, target_data = make_data(S)
	# time dim should be same for input and target
	@assert size(input_data,2) == size(target_data,1)
	esn = make_esn(S, input_data)			#build ESN struct
	mach, y_train, y_test = train_p(S, input_data, target_data, esn)
	target_train, target_test = split_train_test(target_data, S.train_frac);
	plot_train_test(target_train, target_test, y_train, y_test)
	input_data, esn, mach, target_train, target_test, y_train, y_test
end

function plot_train_test(target_train, target_test, y_train, y_test)
	pl=plot(size=(1500,900),layout=(2,1),legend=:none)
	obs = Int(round((2/3)*length(target_test)))
	ind_train = length(target_train)-obs:length(target_train)
	ind_test = length(target_test)-obs:length(target_test)
	time_train = 0.01*ind_train
	time_test = collect(0.01*ind_test) .+ maximum(time_train)
	plot!(time_train,target_train[ind_train])
	plot!(time_train,y_train[ind_train])
	plot!(time_test,target_test[ind_test],subplot=2)
	plot!(time_test,y_test[ind_test],subplot=2)
	display(pl)
end

function train_p(S, input_data, target_data, esn)
	mach, y_train, y_test = brr_fit_tune(esn.states, target_data; train_frac=S.train_frac)
end

end # module PPredict
