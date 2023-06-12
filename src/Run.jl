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

# run exp
using JLD2
run_exp()
d = read_data();
jldsave("/Users/steve/Desktop/d.jld2"; d)

# if already saved, can reload
d=load("/Users/steve/Desktop/d.jld2")["d"];

pl=plot_by_trt(d;show_points=false)

# save fig
using Plots
savefig(pl, "/Users/steve/Desktop/fig.pdf")

# sort params from factorial design
using Printf
function calc_ly_alt()
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

# calc distribution of r2_test values for given set of parameters
# N=5 and F=8.75 gives dbl close to 1.0, use shift=1.0, T=20000.0
# check for res = [25 50 100], sample of n=20
# returns matrix with rows as samples and cols as res_size values
using Serialization
f="/Users/steve/sim/zzOtherLang/julia/projects/Precise/PPredict/output/00_r2_distn"
r2_distn_orig = PPredict.distn_r2_test(seed=7019109242897849223,n=20,T=20000.0,
				res_size=[25 50 100]);
serialize(f,r2_distn_orig);
# recall data
r2_distn = deserialize(f);
using Statistics
for i in 1:size(r2_distn)[2]
	println(minimum(r2_distn[:,i]), " ", maximum(r2_distn[:,i]))
	println(mean(r2_distn[:,i]), " ", std(r2_distn[:,i]))
	println()
end
for i in 1:size(r2_distn)[2]
	println(minimum(r2_distn[:,i]), " ", median(r2_distn[:,i]), " ", 
			maximum(r2_distn[:,i]))
	println()
end

# make plot of lyapunov exponent vs F for N=5
# increasing mult gives more accurate result and smooths plot
# decreasing incr gives greater resolution
incr=0.01
F, ly = calc_ly(4.8:incr:18; mult=4.0);
pl=plot(F,ly,legend=false,xlabel="F",ylabel="Lyapunov exponent",xticks=5:3:17)
savefig(pl,"/Users/steve/Desktop/ly.pdf")

# timing benchmark for a single run, use different shift values
using BenchmarkTools
@btime PPredict.run_exp_timing(res_size=[50],shift=[0.50],T=20000.0)

# get doubling values used in figure
fff=PPredict.rng(0.01,12,4)*7.8;
PPredict.ly2doubling.(lorenz96_ly.(5,fff))

# make figure analyzing effect of increasing res_size for various parameter combos
# full experiment
run_exp(F=[8.116711278 11.840949198],res_size=[25 50 100 200 400 800],
			shift=[1.0 2.0],T=20000.0)
# break into 4 parts to run on different machines for speedup
run_exp(F=[8.116711278],res_size=[25 50 100 200 400 800],	# local
			shift=[1.0],T=20000.0)
run_exp(F=[8.116711278],res_size=[25 50 100 200 400 800],	# alice
			shift=[2.0],T=20000.0)
run_exp(F=[11.840949198],res_size=[25 50 100 200 400 800],	# fisher
			shift=[1.0],T=20000.0)
run_exp(F=[11.840949198],res_size=[25 50 100 200 400 800],	# local round 2
			shift=[2.0],T=20000.0)

# collect data for R2_ts by hand from pdf plot for each parameter combination
# first row is res_size
# second row is F=8.12,  dbl = 1.42, shift = 1.0
# third  row is F=8.12,  dbl = 1.42, shift = 2.0
# fourth row is F=11.84, dbl = 0.52, shift = 1.0
# fifth  row is F=11.84, dbl = 0.52, shift = 2.0
rsize = [25   50   100  200  400  800;
         73.6 77.5 81.7 86.8 90.5 94.0;
         58.5 65.5 70.7 77.0 80.9 84.5;
         24.0 28.5 35.3 38.9 44.6 48.1;
         11.5 14.2 14.7 18.8 22.0 24.6];
xtck = [25;50;100;200;400;800];
wdth = 1.5;
clr = PPredict.Analysis.mma;
pl=plot(rsize[1,:], rsize[2,:], color=clr[3], label="dbl_hi, shift_lo",
			xscale=:log2, yrange=(10,100), xticks=(xtck,string.(xtck)), linewidth=wdth,
			xlabel="Reservoir size", ylabel="R-squared", legend_position=:topleft)
plot!(rsize[1,:], rsize[3,:], color=clr[4], label="dbl_hi, shift_hi", linewidth=wdth)
plot!(rsize[1,:], rsize[4,:], color=clr[3], label="dbl_lo, shift_lo", linewidth=wdth,
		style=:dash)
plot!(rsize[1,:], rsize[5,:], color=clr[4], label="dbl_lo, shift_hi", linewidth=wdth,
		style=:dash)

savefig(pl, "/Users/steve/Desktop/res_size.pdf")
