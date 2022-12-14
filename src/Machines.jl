module Machines
using DataFrames, MLJ, MLJScikitLearnInterface, MLJParticleSwarmOptimization, Printf,
		Suppressor
export brr_fit_tune, best, split_train_test

# esn states come with rows as features and columns as samples
# mlj requires transpose of this setup
matrix_to_dataframe(x::Matrix{Float64}; transpse=true) = 
		DataFrame(transpse ? transpose(x) : x, :auto)

#= BayesianRidgeRegressor hyperparameters, see
https://scikit-learn.org/stable/auto_examples/linear_model/plot_bayesian_ridge_curvefit.html#sphx-glr-auto-examples-linear-model-plot-bayesian-ridge-curvefit-py

default values noted
Int, :n_iter (300)
Float64, :tol(1e-3), :alpha_1, :alpha_2, :lambda_1, :lambda_2 [1e-6 for rest]
Boolean, :compute_score, :fit_intercept, :normalize, :copy_X, :verbose (false)
=#
function brr_fit_tune(states::AbstractArray{Float64}, target::AbstractArray{Float64};
				train_frac=0.7)
	brr = BayesianRidgeRegressor()
	X = matrix_to_dataframe(states)
	y = target
	#println("type of X = ", scitype(X))	# should be Table{AbstractVector{Continuous}}
	#println("type of y = ", scitype(y))	# should be AbstractVector{Continuous}
	# may not be useful to use ensembles for boosted trees
	# brr = EnsembleModel(model=brr, n=3)
	# if using EnsembleModel, then hyperparameters are given as :(model.hyperparameter)
	# add scale=:log10 for log scaling
	lower = 1e-8
	upper = 1e0
	r1 = range(brr, :alpha_1,  lower=lower, upper=upper, scale=:log10)
	r2 = range(brr, :alpha_2,  lower=lower, upper=upper, scale=:log10)
	r3 = range(brr, :lambda_1, lower=lower, upper=upper, scale=:log10)
	r4 = range(brr, :lambda_2, lower=lower, upper=upper, scale=:log10)
	# operation=predict_mode causes predictions to be deterministic {0,1} instead
	# of probabilities on [0,1]
	brr_tuned = TunedModel(model=brr,
				  tuning=AdaptiveParticleSwarm(n_particles=30),	# Grid(resolution=4)
				  resampling=CV(nfolds=6),
				  ranges=[r1,r2,r3,r4],
				  measure=RootMeanSquaredError(),	# adjust if necessary
				  acceleration=default_resource())	# CPUThreads() if not python wrapping
    mach = machine(brr_tuned, X, y)
	if train_frac < 1.0
		train, test = partition(eachindex(y), train_frac) 	# train_frac:1-train_frac split
	else
		train = eachindex(y)
	end
	println("\nFitting brrboost model ...\n")
	@suppress begin
		fit!(mach, rows=train)
	end
	y_train = predict(mach, X[train,:])
	y_test = train_frac < 1.0 ? predict(mach, X[test,:]) : nothing
	return mach, y_train, y_test
end

function split_train_test(data, train_frac)
	index_train, index_test = partition(eachindex(data), train_frac);
	train = data[index_train];
	test = data[index_test];
	return train, test
end

# print hyperparameters
function best(mach; show_all=false)
	bm = report(mach).best_model
	@printf "%-13s = %9.3e\n" "alpha_1" bm.alpha_1
	@printf "%-13s = %9.3e\n" "alpha_2" bm.alpha_2
	@printf "%-13s = %9.3e\n" "lambda_1" bm.lambda_1
	@printf "%-13s = %9.3e\n" "lambda_2" bm.lambda_2
	if show_all
		@printf "%13s = %5d\n" "n_iter" bm.n_iter
		@printf "%-13s = %5.3f\n" "tol" bm.tol
		@printf "%-13s = %-5s\n" "compute_score" bm.compute_score
		@printf "%-13s = %-5s\n" "fit_intercept" bm.fit_intercept
		@printf "%-13s = %-5s\n" "normalize" bm.normalize
		@printf "%-13s = %-5s\n" "copy_X" bm.copy_X
		@printf "%-13s = %-5s\n" "verbose" bm.verbose
	end
end

end # module Machines
