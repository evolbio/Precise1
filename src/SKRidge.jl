# cannot get std for BayesianRidgeRegressor from MLJ interface, so after getting
# hyperparameters for model, redo fit and predict via ScikitLearn to get std
# See https://github.com/cstjean/ScikitLearn.jl/issues/117#issuecomment-1354416770
# Put in separate module to isolate calls to ScikitLearn from those in Machines.jl
# via MLJ interface

module SKRidge
import ScikitLearn as sk
import PyCall: PyNULL
using Suppressor

const BayesianRidge = PyNULL()

function __init__()
	@suppress begin
		@eval sk.@sk_import linear_model: BayesianRidge
	end
end

# https://discourse.julialang.org/t/46131/6
# https://github.com/cstjean/ScikitLearn.jl/issues/50#issuecomment-552071060
function sk_brr(bm, X_train, X_test, y_train)
	reg = BayesianRidge(alpha_1=bm.alpha_1, alpha_2=bm.alpha_2,
 							lambda_1=bm.lambda_1,lambda_2=bm.lambda_2)
	sk.fit!(reg, X_train, y_train)
	y_train, y_train_std = sk.predict(reg, X_train, return_std=true)
	y_test, y_test_std = sk.predict(reg, X_test, return_std=true)
	y_train, y_train_std, y_test, y_test_std
end

end # module SKRidge
