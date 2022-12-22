using PPredict

S = Params(F=5.5, res_size=300, shift=0.5);
S = Params(F=5.5, res_size=30, shift=0.5, T=1000.0);

S = Params(F=5.5, res_size=50, shift=2.0, T=5000.0);
S = Params(F=5.7, res_size=30, shift=2.0, T=5000.0);

d = predict_driver(S);

d = predict_driver(S; save_fig="/Users/steve/Desktop/fig.pdf");

best(d.mach)
r2_percent_print(target_train,y_train,target_test,y_test)
println("Lyapunov exp = ", lorenz96_ly(S.N, S.F))

# sort params from factorial design
using Printf
function calc_ly()
	S_exp = PPredict.exp_param(;N=[10], F=4.7*PPredict.rng(0.005,25,4))
	S_unique = union([(N=x.N, F=x.F) for x in S_exp])
	result = []
	Threads.@threads for s in S_unique
		push!(result, (N=s.N, F=s.F, ly=lorenz96_ly(s.N,s.F)))
	end
	sort!(result, by = x -> x.ly)
	for s in result
		d = (abs(s.ly) < 1e-2 ? 0.0 : log(2)/s.ly)
		@printf "N=%2d F=%5.2f ly=%5.2f d=%5.2f\n" s.N s.F s.ly d
	end
end

# quick plot of Lorenz96
using Plots
S=Params(N=125, F=4.0, T=10.0, fixed_init=false);
trj = PPredict.lorenz96(S);
pl=plot(size=(1500,900),legend=:none)
for i in 1:3 plot!(trj[:,i]) end
display(pl)
