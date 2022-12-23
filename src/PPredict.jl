module PPredict
using ReservoirComputing, DynamicalSystems, Plots, JLD2, MLJ, Dates, Printf
include("Settings.jl")
using .Settings
include("Machines.jl")
using .Machines
include("Analysis.jl")
using .Analysis
export Params
export best, r2_percent, r2_percent_print
export predict_driver, lorenz96_ly, data, stats, calc_stats, run_exp
export read_data, split_data, plot_by_trt, split_shift

####################################################################
# colors, see MMAColors.jl in my private modules

mma = [RGB(0.3684,0.50678,0.7098),RGB(0.8807,0.61104,0.14204),
			RGB(0.56018,0.69157,0.19489), RGB(0.92253,0.38563,0.20918)];

####################################################################

struct stats
	ly
	best_model
	r2_train
	r2_test
end

struct data
	S
	input_data
	esn
	mach
	target_train
	target_test
	y_train
	y_train_std
	y_test
	y_test_std
end

function run_exp()
	#S_exp = exp_param(N=[5], res_size=[25], shift=[0.5, 1.0], T=3000.0)
	S_exp = exp_param(N=[5],res_size=[200],T=20000.0)
	start_time = Dates.format(now(),"yyyymmdd_HHMMSS")
	path = "/Users/steve/sim/zzOtherLang/julia/projects/Precise/PPredict/output/" *
					start_time * "_"
	for S in S_exp
		cpath = @sprintf "%s%02d" path S.run_num
		println("running ", cpath)
		exp_data=predict_driver(S; save_fig=cpath * ".pdf")
		#jldsave(cpath*".jld2"; exp_data; compress=true) # with compression, but is slow
		jldsave(cpath*".jld2"; exp_data)
	end
end

function exp_param(;N=[5 10], F=7.8*rng(0.01,12,4), res_size=[25 50 100 200],
				shift = [0.25 0.5 1.0 2.0], T=5000.0)
	S_exp = [Params(N=i, F=j, res_size=k, shift=m, T=T)
			for i in N, j in F, k in res_size, m in shift]
	for (S,i) in zip(S_exp,1:length(S_exp))
		S_exp[i] = Params(S; run_num=i)
	end
	return S_exp
end

function calc_stats(S, mach, target_train, y_train, target_test, y_test)
	ly = lorenz96_ly(S.N, S.F)
	best_model = report(mach).best_model
	r2_train, r2_test = r2_percent(target_train, y_train, target_test, y_test)
	stats(ly, best_model, r2_train, r2_test)
end

# see https://en.wikipedia.org/wiki/Lorenz_96_model
# The Lorenz-96 model is predefined in DynamicalSystems.jl
# add initial warmup period then cut it off
# add final extra time for shift data, process it later
# trajectory is a Dataset, convert to matrix
function lorenz96(S::Params)
	ds, u = lorenz96_sys(S.N, S.F)
	trj = Matrix(trajectory(ds, S.T+S.warmup+S.shift, u; Δt = S.dt))
	first_ind = Int(round(S.warmup/S.dt)+1)
	return trj[first_ind:end,:]
end

function lorenz96_sys(N, F; fixed_init=false)
	if fixed_init
		u = F * ones(N)
		u[1] += 0.01					# small perturbation 
	else
		u = F * rand(N)
	end
	ds = Systems.lorenz96(N, u; F = F)
	return ds, u
end

function lorenz96_ly(N, F)
	ds, _ = lorenz96_sys(N, F)
	lyapunov(ds, 100000.0, Ttr=1000.0)
end

ly2doubling(ly) = abs(log(2)/ly)
rng_2(b,n) = b .^ (0:n-1)
rng(inc,n,s) = (1:inc:1+(n-1)*inc) .^ s

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

function predict_driver(S; save_fig=nothing)
	input_data, target_data = make_data(S)
	# time dim should be same for input and target
	@assert size(input_data,2) == size(target_data,1)
	esn = make_esn(S, input_data)			#build ESN struct
	mach, y_train, y_train_std, y_test, y_test_std =
						train_p(S, input_data, target_data, esn)
	target_train, target_test = split_train_test(target_data, S.train_frac);
	plot_train_test(S, mach, target_train, target_test, y_train, y_test; save=save_fig)
	data(S, input_data, esn, mach, target_train, target_test, y_train, y_train_std,
						y_test, y_test_std)
end

# scale vectors so that target range is [min,max]
function rescale!(t_train, t_test, y_train, y_test; min=-1.0, max=1.0)
	minv=minimum(vcat(t_train,t_test))
	rngv=maximum(vcat(t_train,t_test)) - minv
	for xvec in (t_train, t_test, y_train, y_test)
		map!(x -> (max-min)*(x-minv)/rngv + min, xvec, xvec)
	end
	return nothing
end

function plot_train_test(S, mach, target_train, target_test, y_train, y_test;
			save=nothing)
	rescale!(target_train, target_test, y_train, y_test)
	obs1 = Int(round((2/3)*length(target_test)))
	obs2 = 2000
	obs = obs1 > obs2 ? obs2 : obs1
	ind_train = length(target_train)-obs:length(target_train)
	ind_test = length(target_test)-obs:length(target_test)
	time_train = 0.01*ind_train
	time_test = collect(0.01*ind_test) .+ maximum(time_train)
	maxv=maximum(vcat(target_train[ind_train], target_test[ind_test],
							y_train[ind_train], y_test[ind_test]))
	minv=minimum(vcat(target_train[ind_train], target_test[ind_test],
							y_train[ind_train], y_test[ind_test]))
	maxp=maximum(abs.([minv,maxv]))
	pl=plot(size=(1500,900),layout=(2,1),legend=:none,yrange=(-maxp,maxp))
	plot!(time_train,target_train[ind_train], color=mma[1], linewidth=2)
	plot!(time_train,y_train[ind_train], color=mma[2], linewidth=2)
	plot!(time_test,target_test[ind_test], color=mma[1], linewidth=2, subplot=2)
	plot!(time_test,y_test[ind_test], color=mma[2], linewidth=2, subplot=2)
	
	st = calc_stats(S, mach, target_train, y_train, target_test, y_test)
	x = mean(time_train)
	y = 0.95*maxp
 	txt1 = @sprintf("N = %2d, F = %4.2f, dbl = %4.2f", S.N, S.F, ly2doubling(st.ly))
 	txt2 = @sprintf("res = %3d, shift = %4.2f", S.res_size, S.shift)
 	txt = @sprintf("%s    %s    R2_tr = %4.1f%1s, R2_ts = %4.1f%1s",
 					txt1, txt2, st.r2_train, "%", st.r2_test, "%")
 	annotate!(x, y, (txt,12, :center, :center), subplot=1)
	save == nothing ? display(pl) : savefig(pl, save)
end

function train_p(S, input_data, target_data, esn)
	mach, y_train, y_train_std, y_test, y_test_std =
				brr_fit_tune(esn.states, target_data; train_frac=S.train_frac)
end

end # module PPredict
